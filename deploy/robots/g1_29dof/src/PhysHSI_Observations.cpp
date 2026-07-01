// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// PhysHSI 特有 Observation 函数注册
//
// 两个特权观测, 数据来源: PrivilegedSubscriber (mqueue topic ← unitree_mujoco)
//
//   end_effector_pos (15维) — 5 个末端在本体坐标系中的位置
//   task_obs         (15维) — 任务感知 (箱子 + 目标点)
//
// 与原始训练代码严格一致:
//   - quat_rotate_inverse(q, v) = q.conjugate() * v
//   - quat_to_tan_norm(q)       = R的前两列按行主序展平 [r00,r01,r10,r11,r20,r21]

#include "PrivilegedState.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"

// =============================================================================
// 全局特权订阅器 (main.cpp 中初始化)
// =============================================================================
std::shared_ptr<PrivilegedSubscriber> g_privileged_sub = nullptr;


namespace isaaclab
{
namespace mdp
{

// =============================================================================
// 辅助函数: quat_rotate_inverse(q, v_world)
//   对应 Python Isaac Lab: quat_rotate_inverse(q, v)
//   将世界坐标系向量旋转到本体坐标系
//   Eigen: q.conjugate() * v  (效果等同于 inverse rotate)
// =============================================================================
inline Eigen::Vector3f quat_rotate_inverse(const Eigen::Quaternionf& q,
                                            const Eigen::Vector3f& v)
{
    return q.conjugate() * v;
}


// =============================================================================
// 辅助函数: quat_to_tan_norm(q)
//   对应 Python: mat = matrix_from_quat(q); return mat[:, :2].reshape(-1)
//   将四元数转为 6D 旋转表示 (Zhou et al., 2019)
//
//   PyTorch mat[:, :2] 取前两列 → reshape 行主序:
//     [r00, r01, r10, r11, r20, r21]
//
//   注意: Eigen 默认列主序, 需要手动按行主序构造成与 Python 一致
// =============================================================================
inline Eigen::Matrix<float, 6, 1> quat_to_tan_norm(const Eigen::Quaternionf& q)
{
    Eigen::Matrix3f R = q.normalized().toRotationMatrix();
    Eigen::Matrix<float, 6, 1> result;
    // 行主序: [R(0,0), R(0,1), R(1,0), R(1,1), R(2,0), R(2,1)]
    result << R(0, 0), R(0, 1), R(1, 0), R(1, 1), R(2, 0), R(2, 1);
    return result;
}


// =============================================================================
// REGISTER_OBSERVATION: end_effector_pos (15 维)
//
// 5 个末端在本体坐标系中的位置, 每末端 3 维:
//   [0-2]   left_palm_link      (左手掌)
//   [3-5]   right_palm_link     (右手掌)
//   [6-8]   left_ankle_pitch_link   (左脚)
//   [9-11]  right_ankle_pitch_link  (右脚)
//   [12-14] d455_link           (头部相机)
//
// 计算: local_pos = quat_rotate_inverse(root_quat, pos_w - root_pos_w)
//       pos_w 和 root_pos_w 来自特权信息话题
// =============================================================================
REGISTER_OBSERVATION(end_effector_pos)
{
    auto& robot = env->robot;
    auto root_quat = robot->data.root_quat_w;

    std::vector<float> result(15, 0.0f);

    if (g_privileged_sub && g_privileged_sub->is_connected())
    {
        auto msg = g_privileged_sub->get_state();
        Eigen::Vector3f root_pos_w(msg.root_pos_w);

        for (int i = 0; i < 5; ++i)
        {
            // 从消息中提取末端世界坐标
            Eigen::Vector3f pos_w(msg.end_effector_pos_w[i * 3 + 0],
                                  msg.end_effector_pos_w[i * 3 + 1],
                                  msg.end_effector_pos_w[i * 3 + 2]);

            // 转换到本体坐标系
            Eigen::Vector3f pos_b = quat_rotate_inverse(root_quat,
                                                         pos_w - root_pos_w);

            result[i * 3 + 0] = pos_b.x();
            result[i * 3 + 1] = pos_b.y();
            result[i * 3 + 2] = pos_b.z();
        }
    }
    // 无特权信息时返回零向量 (策略可能表现下降但不崩溃)

    return result;
}


// =============================================================================
// REGISTER_OBSERVATION: task_obs (15 维)
//
// 组成:
//   [0-2]   box_pos_local     箱子在本体坐标系中的位置
//           quat_rotate_inverse(root_quat, box_pos_w - root_pos_w)
//   [3-8]   box_rot_6d_local  箱子在本体坐标系中的 6D 旋转表示
//           quat_to_tan_norm(conj(root_quat) * box_quat_w)
//   [9-11]  box_size          箱子半边长 [width, depth, height] (静态值)
//   [12-14] goal_pos_local    目标点在本体坐标系中的位置
//           quat_rotate_inverse(root_quat, goal_pos_w - root_pos_w)
//
// 所有数据来自特权信息话题 (unitree_mujoco 发布)
// =============================================================================
REGISTER_OBSERVATION(task_obs)
{
    auto& robot = env->robot;
    auto root_quat = robot->data.root_quat_w;

    std::vector<float> result(15, 0.0f);

    if (g_privileged_sub && g_privileged_sub->is_connected())
    {
        auto msg = g_privileged_sub->get_state();
        Eigen::Vector3f root_pos_w(msg.root_pos_w);

        // ---- (A) box_pos_local (3维) ----
        {
            Eigen::Vector3f box_pos_w(msg.box_pos_w);
            Eigen::Vector3f box_b = quat_rotate_inverse(root_quat,
                                                         box_pos_w - root_pos_w);
            result[0] = box_b.x();
            result[1] = box_b.y();
            result[2] = box_b.z();
        }

        // ---- (B) box_rot_6d_local (6维) ----
        {
            // msg.box_quat_w = [w, x, y, z] (与训练代码一致)
            Eigen::Quaternionf box_quat_w(msg.box_quat_w[0],  // w
                                          msg.box_quat_w[1],  // x
                                          msg.box_quat_w[2],  // y
                                          msg.box_quat_w[3]); // z

            // 箱子在本体坐标系的姿态: conj(root_quat) * box_quat_w
            Eigen::Quaternionf box_quat_b = root_quat.conjugate() * box_quat_w;

            Eigen::Matrix<float, 6, 1> r6d = quat_to_tan_norm(box_quat_b);
            for (int i = 0; i < 6; ++i)
                result[3 + i] = r6d(i);
        }

        // ---- (C) box_size (3维, 静态值) ----
        {
            result[9]  = msg.box_size[0];  // width (half)
            result[10] = msg.box_size[1];  // depth (half)
            result[11] = msg.box_size[2];  // height (half)
        }

        // ---- (D) goal_pos_local (3维) ----
        {
            Eigen::Vector3f goal_pos_w(msg.goal_pos_w);
            Eigen::Vector3f goal_b = quat_rotate_inverse(root_quat,
                                                          goal_pos_w - root_pos_w);
            result[12] = goal_b.x();
            result[13] = goal_b.y();
            result[14] = goal_b.z();
        }
    }

    return result;
}

} // namespace mdp
} // namespace isaaclab
