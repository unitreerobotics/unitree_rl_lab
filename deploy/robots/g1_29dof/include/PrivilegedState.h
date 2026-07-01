// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// 特权信息 DDS Topic 订阅器
//
// 话题名称: rt/privileged_state
// 传输层:   Cyclone DDS (与 LowState/LowCmd 同一 DDS 总线)
// 消息类型: PrivilegedStateMsg (128 bytes, POD)
//
// 架构:
//   unitree_mujoco (DDS DataWriter) ──DDS 域 0──→ g1_ctrl (DDS DataReader)
//
// 与现有架构一致:
//   - 同一 DDS 域 (domain 0), 由 unitree_sdk2 ChannelFactory 初始化
//   - 话题命名遵循 rt/ 约定 (与 rt/lowstate 等并行)
//   - 消息格式为固定大小 POD struct, DDS 自动序列化

#pragma once

#include <eigen3/Eigen/Dense>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

// Cyclone DDS C++11 API (已链接 ddscxx)
#include <dds/dds.hpp>

#include <spdlog/spdlog.h>


// =============================================================================
// PrivilegedStateMsg — DDS 话题消息体 (128 bytes, POD)
//
// 遵循 DDS topic type 要求:
//   - 纯 POD (trivially copyable)
//   - 无指针, 无动态数组
//   - 固定大小 (编译期可知)
// =============================================================================

struct PrivilegedStateMsg
{
    // ---- 末端执行器世界坐标 (5 bodies × 3 = 15 floats) ----
    float end_effector_pos_w[15];

    // ---- 机器人根Link世界坐标 (3 floats) ----
    float root_pos_w[3];

    // ---- 任务信息 ----
    float box_pos_w[3];
    float box_quat_w[4];     // w, x, y, z
    float box_size[3];       // half-size [w, d, h]
    float goal_pos_w[3];

    // ---- 序列号 ----
    uint32_t sequence;
    uint32_t padding;        // 对齐
};

static_assert(sizeof(PrivilegedStateMsg) == 128,
              "PrivilegedStateMsg must be exactly 128 bytes");


// =============================================================================
// PrivilegedSubscriber — 特权信息 DDS 订阅器
//
// 基于 Cyclone DDS DataReader, 与 unitree_sdk2 共享 DDS 域 0。
// 使用 WaitSet 实现非阻塞等待: 有消息时读取, 无消息时立即返回。
//
// 使用方式 (与 LowState_t 订阅模式一致):
//   auto sub = std::make_shared<PrivilegedSubscriber>();
//   sub->connect();           // 等待 DDS 发现 publisher
//   sub->receive();           // 每周期拉取 (非阻塞)
//   auto msg = sub->get_state();  // 获取最新消息
// =============================================================================

class PrivilegedSubscriber
{
public:
    static constexpr const char* TOPIC_NAME = "rt/privileged_state";
    static constexpr int DOMAIN_ID = 0;  // 与 ChannelFactory 同一域

    PrivilegedSubscriber(const std::string& topic_name = TOPIC_NAME,
                         int domain_id = DOMAIN_ID)
        : topic_name_(topic_name)
        , domain_id_(domain_id)
    {}

    ~PrivilegedSubscriber() { disconnect(); }

    // ---- 订阅话题 (类似 DDS DataReader 创建 + wait_for_connection) ----
    bool connect(int timeout_ms = 5000)
    {
        try
        {
            // 在 DDS 域 0 创建 participant (与 unitree_sdk2 同一域)
            participant_ = dds::domain::DomainParticipant(domain_id_);

            // 创建话题 (Cyclone DDS 自动注册 POD 类型)
            topic_ = dds::topic::Topic<PrivilegedStateMsg>(
                participant_, topic_name_);

            // 创建 Subscriber + DataReader
            dds::sub::Subscriber sub(participant_);
            dds::sub::qos::DataReaderQos dr_qos;
            dr_qos << dds::core::policy::Reliability::Reliable();
            dr_qos << dds::core::policy::Durability::TransientLocal();
            dr_qos << dds::core::policy::History::KeepLast(10);

            reader_ = dds::sub::DataReader<PrivilegedStateMsg>(
                sub, topic_, dr_qos);

            // 创建 WaitSet + ReadCondition (非阻塞读取)
            waitset_ = dds::core::cond::WaitSet();
            dds::sub::cond::ReadCondition rc(
                reader_,
                dds::sub::status::DataState::any(),
                [](const PrivilegedStateMsg&) { return true; }
            );
            waitset_.attach_condition(rc);

            // 等待 publisher 上线 (通过 DDS 发现机制)
            int elapsed = 0;
            const int poll_ms = 100;
            while (elapsed < timeout_ms)
            {
                // 检查是否有匹配的 DataWriter
                auto pub_count = dds::sub::matched_publications(reader_);
                // 注意: dds::sub::matched_publications 返回
                //       dds::core::InstanceHandleSeq
                if (pub_count.size() > 0)
                {
                    spdlog::info(
                        "PrivilegedSubscriber: discovered publisher on "
                        "topic '{}' (domain {})",
                        topic_name_, domain_id_);
                    connected_ = true;
                    return true;
                }

                usleep(poll_ms * 1000);
                elapsed += poll_ms;
            }

            spdlog::warn("PrivilegedSubscriber: no publisher for topic '{}' "
                         "within {}ms", topic_name_, timeout_ms);
            return false;
        }
        catch (const dds::core::Exception& e)
        {
            spdlog::error("PrivilegedSubscriber: DDS error — {}", e.what());
            disconnect();
            return false;
        }
    }

    // ---- 接收消息 (非阻塞, 类似 DDS take) ----
    // 在主循环或 FSM pre_run 中周期调用
    void receive()
    {
        if (!connected_) return;

        try
        {
            // 检查是否有数据 (超时 0 = 非阻塞)
            auto conditions = waitset_.wait(dds::core::Duration(0, 0));

            if (conditions.size() > 0)
            {
                // 读取所有可用样本, 保留最新一条
                auto samples = reader_.take();
                if (samples.length() > 0)
                {
                    // 取最后一条 (最新消息)
                    std::lock_guard<std::mutex> lk(mutex_);
                    cached_msg_ = samples[samples.length() - 1].data();
                    msg_count_ += samples.length();
                }
            }
        }
        catch (const dds::core::Exception& e)
        {
            spdlog::warn("PrivilegedSubscriber: receive error — {}", e.what());
        }
    }

    // ---- 取消订阅 ----
    void disconnect()
    {
        connected_ = false;
        // DDS 对象在析构时自动清理
        reader_ = dds::sub::DataReader<PrivilegedStateMsg>(dds::core::null);
        topic_ = dds::topic::Topic<PrivilegedStateMsg>(dds::core::null);
        participant_ = dds::domain::DomainParticipant(dds::core::null);
    }

    // ---- 获取最新消息 (线程安全) ----
    PrivilegedStateMsg get_state() const
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return cached_msg_;
    }

    bool is_connected() const { return connected_; }
    uint64_t msg_count() const { return msg_count_; }

private:
    std::string topic_name_;
    int domain_id_;

    dds::domain::DomainParticipant participant_{dds::core::null};
    dds::topic::Topic<PrivilegedStateMsg> topic_{dds::core::null};
    dds::sub::DataReader<PrivilegedStateMsg> reader_{dds::core::null};
    dds::core::cond::WaitSet waitset_;

    bool connected_ = false;
    PrivilegedStateMsg cached_msg_{};
    uint64_t msg_count_ = 0;
    mutable std::mutex mutex_;
};


// =============================================================================
// 全局指针 (在 main.cpp 中定义和初始化)
// =============================================================================
extern std::shared_ptr<PrivilegedSubscriber> g_privileged_sub;


// =============================================================================
// PrivilegedPublisher — 特权信息 DDS 发布器 (unitree_mujoco 侧)
//
// 基于 Cyclone DDS DataWriter。
//
// 使用方式:
//   auto pub = std::make_shared<PrivilegedPublisher>();
//   pub->connect();
//   pub->publish(msg);  // 每仿真步调用
// =============================================================================

class PrivilegedPublisher
{
public:
    static constexpr const char* TOPIC_NAME = "rt/privileged_state";
    static constexpr int DOMAIN_ID = 0;

    PrivilegedPublisher(const std::string& topic_name = TOPIC_NAME,
                        int domain_id = DOMAIN_ID)
        : topic_name_(topic_name)
        , domain_id_(domain_id)
    {}

    ~PrivilegedPublisher() { disconnect(); }

    // 创建 DDS DataWriter
    bool connect()
    {
        try
        {
            participant_ = dds::domain::DomainParticipant(domain_id_);

            topic_ = dds::topic::Topic<PrivilegedStateMsg>(
                participant_, topic_name_);

            dds::pub::Publisher pub(participant_);

            dds::pub::qos::DataWriterQos dw_qos;
            dw_qos << dds::core::policy::Reliability::Reliable();
            dw_qos << dds::core::policy::Durability::TransientLocal();
            dw_qos << dds::core::policy::History::KeepLast(10);

            writer_ = dds::pub::DataWriter<PrivilegedStateMsg>(
                pub, topic_, dw_qos);

            spdlog::info("PrivilegedPublisher: publishing to topic '{}' "
                         "(domain {})", topic_name_, domain_id_);
            return true;
        }
        catch (const dds::core::Exception& e)
        {
            spdlog::error("PrivilegedPublisher: DDS error — {}", e.what());
            return false;
        }
    }

    // 发布消息
    void publish(const PrivilegedStateMsg& msg)
    {
        try
        {
            writer_.write(msg);
        }
        catch (const dds::core::Exception& e)
        {
            spdlog::warn("PrivilegedPublisher: write error — {}", e.what());
        }
    }

    void disconnect()
    {
        writer_ = dds::pub::DataWriter<PrivilegedStateMsg>(dds::core::null);
        topic_ = dds::topic::Topic<PrivilegedStateMsg>(dds::core::null);
        participant_ = dds::domain::DomainParticipant(dds::core::null);
    }

private:
    std::string topic_name_;
    int domain_id_;

    dds::domain::DomainParticipant participant_{dds::core::null};
    dds::topic::Topic<PrivilegedStateMsg> topic_{dds::core::null};
    dds::pub::DataWriter<PrivilegedStateMsg> writer_{dds::core::null};
};
