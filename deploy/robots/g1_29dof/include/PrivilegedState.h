// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// 特权信息 topic 订阅器 — 基于 POSIX 消息队列 (mqueue)
//
// 话题名称: /physhsi_privileged_state
// 发布端:   unitree_mujoco (每个仿真步发送一次)
// 订阅端:   g1_ctrl (PrivilegedSubscriber 周期性接收)
//
// 消息队列特点:
//   - 命名话题 (mq_open 按名称打开, 类似 DDS topic)
//   - 发布/订阅 (mq_send / mq_receive)
//   - 非阻塞接收 (O_NONBLOCK, 不阻塞 g1_ctrl 主循环)
//   - 内核持久 (mq_unlink 前一直存在)
//   - 消息优先级支持 (最新消息用高优先级)

#pragma once

#include <eigen3/Eigen/Dense>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <memory>
#include <vector>

#include <mqueue.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

#include <spdlog/spdlog.h>


// =============================================================================
// PrivilegedStateMsg — 特权信息消息体 (128 bytes, 固定大小)
//
// 总大小 = 15×4 + 3×4 + 3×4 + 4×4 + 3×4 + 3×4 = 124 bytes
// 与 mqueue msgsize 严格对齐
// =============================================================================

struct PrivilegedStateMsg
{
    // ---- 末端执行器世界坐标 (5 bodies × 3 = 15 floats, 60 bytes) ----
    // 顺序: [0-2]=left_palm, [3-5]=right_palm,
    //        [6-8]=left_ankle_pitch, [9-11]=right_ankle_pitch, [12-14]=d455_link
    float end_effector_pos_w[15];

    // ---- 机器人根Link世界坐标 (3 floats, 12 bytes) ----
    float root_pos_w[3];

    // ---- 任务信息 ----
    float box_pos_w[3];       // 箱子世界坐标 (12 bytes)
    float box_quat_w[4];      // 箱子姿态四元数 wxyz (16 bytes)
    float box_size[3];        // 箱子半边长 [w, d, h] (12 bytes)
    float goal_pos_w[3];      // 目标点世界坐标 (12 bytes)

    // ---- 序列号 (可选, 用于检测丢帧) ----
    uint32_t sequence;
    uint32_t padding;         // 对齐到 128 bytes
};

// 编译期验证
static_assert(sizeof(PrivilegedStateMsg) == 128,
              "PrivilegedStateMsg must be exactly 128 bytes for mqueue msgsize");


// =============================================================================
// PrivilegedSubscriber — 特权信息话题订阅器
//
// 从 POSIX 消息队列 "话题" 接收 unitree_mujoco 发布的特权状态。
// 线程安全: receive() 和 get_state() 可在不同线程调用。
//
// 使用方式 (同 DDS 订阅模式):
//   auto sub = std::make_shared<PrivilegedSubscriber>();
//   sub->connect();           // 打开话题 (类似 DDS wait_for_connection)
//   sub->receive();           // 主循环中周期性收消息 (类似 DDS read/take)
//   auto state = sub->get_state();  // observation 计算中读取
// =============================================================================

class PrivilegedSubscriber
{
public:
    // 话题名称 (类似 DDS topic name "rt/privileged_state")
    static constexpr const char* TOPIC_NAME = "/physhsi_privileged_state";

    PrivilegedSubscriber(const std::string& topic_name = TOPIC_NAME)
        : topic_name_(topic_name)
    {}

    ~PrivilegedSubscriber() { disconnect(); }

    // ---- 订阅话题 (类似 DDS DataReader 创建) ----
    // timeout_ms: 等待 publisher 创建话题的超时时间
    bool connect(int timeout_ms = 5000)
    {
        int elapsed = 0;
        const int poll_ms = 50;

        while (elapsed < timeout_ms)
        {
            // O_RDONLY | O_NONBLOCK: 只读 + 非阻塞 (不卡住 g1_ctrl 主循环)
            mqd_ = mq_open(topic_name_.c_str(), O_RDONLY | O_NONBLOCK);
            if (mqd_ >= 0)
            {
                // 获取消息队列属性, 验证消息大小匹配
                struct mq_attr attr;
                if (mq_getattr(mqd_, &attr) == 0)
                {
                    if (attr.mq_msgsize == sizeof(PrivilegedStateMsg))
                    {
                        spdlog::info("PrivilegedSubscriber: subscribed to topic '{}' "
                                     "(max_msgs={}, msgsize={})",
                                     topic_name_, attr.mq_maxmsg, attr.mq_msgsize);
                        connected_ = true;
                        return true;
                    }
                    else
                    {
                        spdlog::error("PrivilegedSubscriber: topic msgsize mismatch "
                                      "(expected {}, got {})",
                                      sizeof(PrivilegedStateMsg), attr.mq_msgsize);
                        mq_close(mqd_);
                        mqd_ = -1;
                        return false;
                    }
                }
            }

            if (errno != ENOENT && errno != EAGAIN)
            {
                spdlog::error("PrivilegedSubscriber: mq_open failed (errno={})", errno);
                return false;
            }

            usleep(poll_ms * 1000);
            elapsed += poll_ms;
        }

        spdlog::warn("PrivilegedSubscriber: topic '{}' not available within {}ms",
                     topic_name_, timeout_ms);
        return false;
    }

    // ---- 接收消息 (类似 DDS DataReader::take) ----
    // 非阻塞: 没有新消息时立即返回
    // 高频调用: 建议在主循环或 FSM pre_run 中调用
    void receive()
    {
        if (mqd_ < 0) return;

        PrivilegedStateMsg msg;
        // 非阻塞接收, 取最高优先级消息 (模拟 DDS take 语义)
        ssize_t n = mq_receive(mqd_, reinterpret_cast<char*>(&msg),
                               sizeof(PrivilegedStateMsg), nullptr);

        if (n == sizeof(PrivilegedStateMsg))
        {
            std::lock_guard<std::mutex> lk(mutex_);
            cached_msg_ = msg;
            msg_count_++;
        }
        // EAGAIN: 无消息 (正常, 非阻塞模式)
        // 其他错误: 忽略, 下一轮重试
    }

    // 取消订阅
    void disconnect()
    {
        if (mqd_ >= 0)
        {
            mq_close(mqd_);
            mqd_ = -1;
        }
        connected_ = false;
    }

    // ---- 获取最新消息 (线程安全) ----
    PrivilegedStateMsg get_state() const
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return cached_msg_;
    }

    // ---- 状态查询 ----
    bool is_connected() const { return connected_; }
    uint64_t msg_count() const { return msg_count_; }

private:
    std::string topic_name_;
    mqd_t mqd_ = -1;
    bool connected_ = false;

    PrivilegedStateMsg cached_msg_{};
    uint64_t msg_count_ = 0;
    mutable std::mutex mutex_;
};


// =============================================================================
// 全局指针 (在 main.cpp 中初始化)
// =============================================================================
extern std::shared_ptr<PrivilegedSubscriber> g_privileged_sub;


// =============================================================================
// PrivilegedPublisher — 特权信息话题发布器 (unitree_mujoco 侧使用)
//
// 使用方式:
//   auto pub = std::make_shared<PrivilegedPublisher>();
//   pub->connect();
//   pub->publish(msg);  // 每个仿真步调用
// =============================================================================

class PrivilegedPublisher
{
public:
    static constexpr const char* TOPIC_NAME = "/physhsi_privileged_state";

    PrivilegedPublisher(const std::string& topic_name = TOPIC_NAME)
        : topic_name_(topic_name)
    {}

    ~PrivilegedPublisher() { disconnect(); }

    // 创建话题 (类似 DDS DataWriter 创建)
    bool connect(int max_msgs = 10)
    {
        struct mq_attr attr;
        attr.mq_flags   = 0;                          // 阻塞模式 (发布端)
        attr.mq_maxmsg  = max_msgs;                   // 最多缓存 10 条消息
        attr.mq_msgsize = sizeof(PrivilegedStateMsg); // 消息大小 128 bytes
        attr.mq_curmsgs = 0;

        // O_CREAT | O_WRONLY: 创建 + 只写
        mqd_ = mq_open(topic_name_.c_str(),
                       O_CREAT | O_WRONLY,
                       0666,
                       &attr);

        if (mqd_ < 0)
        {
            spdlog::error("PrivilegedPublisher: mq_open failed (errno={})", errno);
            return false;
        }

        spdlog::info("PrivilegedPublisher: publishing to topic '{}'", topic_name_);
        return true;
    }

    // 发布消息 (类似 DDS DataWriter::write)
    void publish(const PrivilegedStateMsg& msg)
    {
        if (mqd_ < 0) return;

        // 优先级 0 (默认), 高优先级消息可以插队
        if (mq_send(mqd_, reinterpret_cast<const char*>(&msg),
                    sizeof(PrivilegedStateMsg), 0) != 0)
        {
            // 队列满时丢弃最旧消息 (通过 mq_setattr 或其他方式)
            // 简单处理: 忽略, 下一帧继续
        }
    }

    void disconnect()
    {
        if (mqd_ >= 0)
        {
            mq_close(mqd_);
            mq_unlink(topic_name_.c_str());  // 删除话题 (类似 DDS delete_topic)
            mqd_ = -1;
        }
    }

private:
    std::string topic_name_;
    mqd_t mqd_ = -1;
};
