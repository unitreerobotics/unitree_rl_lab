// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// PhysHSI 特有 Observation 函数注册
//
// 遵循参考 commit 的模式:
//   CameraData  → REGISTER_OBSERVATION(depth_image) → asset->data.depth_image_buffer
//   TorsoImu    → (通过 CameraArticulation 访问)
//   PrivilegedState → REGISTER_OBSERVATION(end_effector_pos / task_obs)
//                     → FSMState::privilegedstate->msg_  (与 lowstate->msg_ 同模式)
//
// 数据来源: unitree_mujoco 通过 DDS topic rt/privileged_state 发布
// 订阅方式: PrivilegedState_t (继承 SubscriptionBase, 与 LowState_t 相同模式)

#include "FSM/FSMState.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"

namespace isaaclab
{
namespace mdp
{

// =============================================================================
// 辅助函数
// =============================================================================

// quat_rotate_inverse(q, v_world) — 世界坐标转本体坐标
inline Eigen::Vector3f quat_rotate_inverse(const Eigen::Quaternionf& q,
                                            const Eigen::Vector3f& v)
{
    return q.conjugate() * v;
}

// quat_to_tan_norm(q) — 四元数转 6D 旋转表示
// 对应 Python: mat = matrix_from_quat(q); mat[:, :2].reshape(-1)
// 输出顺序: [r00, r01, r10, r11, r20, r21] (行主序, 与 PyTorch 一致)
inline Eigen::Matrix<float, 6, 1> quat_to_tan_norm(const Eigen::Quaternionf& q)
{
    Eigen::Matrix3f R = q.normalized().toRotationMatrix();
    Eigen::Matrix<float, 6, 1> result;
    result << R(0, 0), R(0, 1), R(1, 0), R(1, 1), R(2, 0), R(2, 1);
    return result;
}


// =============================================================================
// REGISTER_OBSERVATION: end_effector_pos (15 维)
//
// 访问模式: FSMState::privilegedstate->msg_  (与 lowstate->msg_ 同模式)
// =============================================================================

REGISTER_OBSERVATION(end_effector_pos)
{
    auto& robot = env->robot;
    auto root_quat = robot->data.root_quat_w;

    std::vector<float> result(15, 0.0f);

    if (FSMState::privilegedstate)
    {
        std::lock_guard<std::mutex> lock(FSMState::privilegedstate->mutex_);
        auto& msg = FSMState::privilegedstate->msg_;

        // 从 DDS 消息中构建 root_pos_w
        Eigen::Vector3f root_pos_w(msg.root_pos_w()[0],
                                   msg.root_pos_w()[1],
                                   msg.root_pos_w()[2]);

        // 5 个末端在本体坐标系中的位置
        for (int i = 0; i < 5; ++i)
        {
            Eigen::Vector3f pos_w(msg.end_effector_pos_w()[i * 3 + 0],
                                  msg.end_effector_pos_w()[i * 3 + 1],
                                  msg.end_effector_pos_w()[i * 3 + 2]);

            Eigen::Vector3f pos_b = quat_rotate_inverse(root_quat,
                                                         pos_w - root_pos_w);

            result[i * 3 + 0] = pos_b.x();
            result[i * 3 + 1] = pos_b.y();
            result[i * 3 + 2] = pos_b.z();
        }
    }

    return result;
}


// =============================================================================
// REGISTER_OBSERVATION: task_obs (15 维)
//
// 访问模式: FSMState::privilegedstate->msg_  (与 lowstate->msg_ 同模式)
// =============================================================================

REGISTER_OBSERVATION(task_obs)
{
    auto& robot = env->robot;
    auto root_quat = robot->data.root_quat_w;

    std::vector<float> result(15, 0.0f);

    if (FSMState::privilegedstate)
    {
        std::lock_guard<std::mutex> lock(FSMState::privilegedstate->mutex_);
        auto& msg = FSMState::privilegedstate->msg_;

        Eigen::Vector3f root_pos_w(msg.root_pos_w()[0],
                                   msg.root_pos_w()[1],
                                   msg.root_pos_w()[2]);

        // ---- (A) box_pos_local (3维) ----
        {
            Eigen::Vector3f box_w(msg.box_pos_w()[0],
                                  msg.box_pos_w()[1],
                                  msg.box_pos_w()[2]);
            Eigen::Vector3f box_b = quat_rotate_inverse(root_quat,
                                                         box_w - root_pos_w);
            result[0] = box_b.x();
            result[1] = box_b.y();
            result[2] = box_b.z();
        }

        // ---- (B) box_rot_6d_local (6维) ----
        {
            Eigen::Quaternionf box_quat_w(msg.box_quat_w()[0],  // w
                                          msg.box_quat_w()[1],  // x
                                          msg.box_quat_w()[2],  // y
                                          msg.box_quat_w()[3]); // z
            Eigen::Quaternionf box_quat_b = root_quat.conjugate() * box_quat_w;
            Eigen::Matrix<float, 6, 1> r6d = quat_to_tan_norm(box_quat_b);
            for (int i = 0; i < 6; ++i)
                result[3 + i] = r6d(i);
        }

        // ---- (C) box_size (3维, 静态) ----
        {
            result[9]  = msg.box_size()[0];
            result[10] = msg.box_size()[1];
            result[11] = msg.box_size()[2];
        }

        // ---- (D) goal_pos_local (3维) ----
        {
            Eigen::Vector3f goal_w(msg.goal_pos_w()[0],
                                   msg.goal_pos_w()[1],
                                   msg.goal_pos_w()[2]);
            Eigen::Vector3f goal_b = quat_rotate_inverse(root_quat,
                                                          goal_w - root_pos_w);
            result[12] = goal_b.x();
            result[13] = goal_b.y();
            result[14] = goal_b.z();
        }
    }

    return result;
}

} // namespace mdp
} // namespace isaaclab
