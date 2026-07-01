// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// 特权信息 DDS 订阅器 — 完全遵循 ros2_sub.h 参考模式
//
// 模式: 继承 SubscriptionBase<MsgType>, 传入 topic 名称
//   参考: CameraData  : SubscriptionBase<sensor_msgs::msg::dds_::PointCloud2_>
//        TorsoImu    : SubscriptionBase<unitree_hg::msg::dds_::IMUState_>
//   新增: PrivilegedState : SubscriptionBase<unitree_hg::msg::dds_::PrivilegedState_>
//
// 消息类型 PrivilegedState_ 的 IDL 定义见下方注释;
// 需在 unitree_sdk2 的 unitree/idl/hg/ 目录下添加 IDL 文件并用 idlc 生成 .hpp
//
// IDL 定义 (unitree/idl/hg/PrivilegedState.idl):
//
//   module unitree_hg {
//     module msg {
//       struct PrivilegedState_ {
//         float end_effector_pos_w[15];  // 5 bodies × 3
//         float root_pos_w[3];
//         float box_pos_w[3];
//         float box_quat_w[4];           // w, x, y, z
//         float box_size[3];
//         float goal_pos_w[3];
//         unsigned long sequence;
//         unsigned long padding;
//       };
//     };
//   };

#ifndef PRIVILEGED_STATE_SUB_H
#define PRIVILEGED_STATE_SUB_H

#include <eigen3/Eigen/Dense>
#include "unitree/dds_wrapper/common/Subscription.h"
#include "unitree/dds_wrapper/robots/g1/defines.h"

// ★ IDL 生成的消息类型头文件 (需要在 unitree_sdk2 中添加 IDL 并编译生成)
#include "unitree/idl/hg/PrivilegedState_.hpp"

namespace unitree
{
namespace robot
{
namespace g1
{
namespace subscription
{

// =============================================================================
// PrivilegedState — 特权信息 DDS 订阅器
//
// 完全遵循 CameraData / TorsoImu 的模式:
//   - 继承 SubscriptionBase<MsgType> (unitree_sdk2 的 DDS 订阅封装)
//   - 构造函数传入 topic 名称
//   - 自动获得 update() / wait_for_connection() / isTimeout() / msg_ / mutex_
//
// 话题: rt/privileged_state
// 类型: unitree_hg::msg::dds_::PrivilegedState_
// 域:   与 ChannelFactory::Init() 指定的同一 DDS 域
// =============================================================================

class PrivilegedState : public SubscriptionBase<unitree_hg::msg::dds_::PrivilegedState_>
{
public:
    using SharedPtr = std::shared_ptr<PrivilegedState>;

    PrivilegedState(std::string topic = "rt/privileged_state")
        : SubscriptionBase<MsgType>(topic)
    {}
};

} // namespace subscription
} // namespace g1
} // namespace robot
} // namespace unitree

#endif // PRIVILEGED_STATE_SUB_H
