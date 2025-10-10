from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.mdp.curriculums import modify_env_param



def modify_force_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple,
    steps_per_level: int = 1000,
    max_level: int = 10,
) -> tuple | object:
    target_level = min(env.common_step_counter // steps_per_level, max_level)
    
    if target_level > old_value[1]:
        new_force_max = float(target_level)
        return (-new_force_max, new_force_max)
    
    return modify_env_param.NO_CHANGE

def modify_torque_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple,
    steps_per_level: int = 1000,
    max_level: int = 10,
) -> tuple | object:
    target_level = min(env.common_step_counter // steps_per_level, max_level)
    
    if target_level > old_value[1]:
        new_torque_max = float(target_level)
        return (-new_torque_max, new_torque_max)
    
    return modify_env_param.NO_CHANGE

def force_torque_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    steps_per_level: int = 1000,
    max_level: int = 10
) -> torch.Tensor:
    target_level = min(env.common_step_counter // steps_per_level, max_level)
    return torch.tensor(target_level, device=env.device)

def push_vel_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    steps_per_level: int = 5000,
    increment: float = 0.05,
    max_speed: float = 1.0,
) -> torch.Tensor:
    max_speed = abs(max_speed)
    level = int(env.common_step_counter // steps_per_level)
    target = float(min(level * increment, max_speed))
    return torch.tensor(target, device=env.device)

def modify_push_velocity_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: dict,
    steps_per_level: int = 5000,
    increment: float = 0.05,
    max_speed: float = 1.0,
) -> dict | object:

    max_speed = abs(max_speed)
    level = int(env.common_step_counter // steps_per_level)
    target = float(min(level * increment, max_speed))

    current_x_max = old_value['x'][1]
    if target > current_x_max:
        return {
            "x": (-target, target),
            "y": (-target, target),
        }
    return modify_env_param.NO_CHANGE

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
