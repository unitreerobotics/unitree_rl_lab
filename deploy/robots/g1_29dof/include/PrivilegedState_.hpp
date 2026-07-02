// =============================================================================
// PrivilegedState_.hpp  — 特权信息 DDS 消息类型定义 (替 idlc 生成)
//
// 来源: idl/PrivilegedState.idl
// 用途: SubscriptionBase<unitree_hg::msg::dds_::PrivilegedState_> 的模板参数
//       经 Cyclone DDS C++11 API 自动序列化，无需 idlc
//
// 原理: Cyclone DDS C++11 对 POD struct 可通过 topic_type_support 特化
//       实现零拷贝序列化 (128 bytes 直传，无动态字段)
// =============================================================================

#ifndef UNITREE_HG_MSG_DDS__PRIVILEGED_STATE__HPP
#define UNITREE_HG_MSG_DDS__PRIVILEGED_STATE__HPP

#include <cstdint>
#include <cstring>

namespace unitree_hg {
namespace msg {
namespace dds_ {

// =============================================================================
// PrivilegedState_ — 128 bytes POD, DDS 兼容
//
// ★ 字段顺序、类型、数组大小必须与 IDL 定义严格一致
//    Cyclone DDS CDR 序列化依赖 struct 内存布局
// =============================================================================

class PrivilegedState_
{
public:
    // ---- 原始数据访问 (符合 IDL 生成代码的访问器命名惯例) ----
    float* end_effector_pos_w()     { return m_end_effector_pos_w; }
    const float* end_effector_pos_w() const { return m_end_effector_pos_w; }

    float* root_pos_w()             { return m_root_pos_w; }
    const float* root_pos_w() const { return m_root_pos_w; }

    float* box_pos_w()             { return m_box_pos_w; }
    const float* box_pos_w() const { return m_box_pos_w; }

    float* box_quat_w()            { return m_box_quat_w; }
    const float* box_quat_w() const { return m_box_quat_w; }

    float* box_size()              { return m_box_size; }
    const float* box_size() const  { return m_box_size; }

    float* goal_pos_w()            { return m_goal_pos_w; }
    const float* goal_pos_w() const { return m_goal_pos_w; }

    uint32_t& sequence()           { return m_sequence; }
    const uint32_t& sequence() const { return m_sequence; }

    uint32_t& padding()            { return m_padding; }
    const uint32_t& padding() const { return m_padding; }

    // DDS 要求: 默认构造函数 zero-initialize
    PrivilegedState_() { std::memset(this, 0, sizeof(PrivilegedState_)); }

private:
    float    m_end_effector_pos_w[15];  // offset 0,  60 bytes
    float    m_root_pos_w[3];           // offset 60, 12 bytes
    float    m_box_pos_w[3];            // offset 72, 12 bytes
    float    m_box_quat_w[4];           // offset 84, 16 bytes
    float    m_box_size[3];             // offset 100,12 bytes
    float    m_goal_pos_w[3];           // offset 112,12 bytes
    uint32_t m_sequence;               // offset 124,4 bytes
    uint32_t m_padding;                // offset 128,4 bytes → total 128
};

// 编译期验证大小
static_assert(sizeof(PrivilegedState_) == 128,
              "PrivilegedState_ must be exactly 128 bytes");
static_assert(alignof(PrivilegedState_) <= 8,
              "PrivilegedState_ alignment must not exceed 8 bytes");

} // namespace dds_
} // namespace msg
} // namespace unitree_hg

// =============================================================================
// DDS 类型注册 — 告诉 Cyclone DDS 这是合法的 topic 类型
//
// Cyclone DDS C++11 通过 dds::topic::is_topic<T> 和
// dds::topic::topic_type_support<T> 两个特化来识别 topic 类型。
//
// 对于纯 POD struct (无指针、无字符串、固定大小),
// Cyclone DDS 使用默认 CDR 序列化: 直接 memcpy 128 bytes,
// 无需手写 serialize/deserialize 函数。
// =============================================================================

#include <dds/dds.hpp>

namespace dds {
namespace topic {

template<>
struct is_topic<unitree_hg::msg::dds_::PrivilegedState_>
{
    static constexpr bool value = true;
};

template<>
struct topic_type_support<unitree_hg::msg::dds_::PrivilegedState_>
{
    // 对于固定大小的 POD 类型，返回类型名用于 DDS 发现匹配
    static const std::string& name()
    {
        static const std::string n =
            "unitree_hg::msg::dds_::PrivilegedState_";
        return n;
    }
};

} // namespace topic
} // namespace dds

#endif // UNITREE_HG_MSG_DDS__PRIVILEGED_STATE__HPP
