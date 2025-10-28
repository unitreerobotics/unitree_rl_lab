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
    max_level: int = 3,
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
    reward_term_name: str = "orientation_exp",
    increment: float = 0.5,
    max_level: float = 3.0,
    performance_thresh: float = 0.97,
) -> torch.Tensor:

    force_event = env.event_manager.get_term_cfg("base_external_force")
    current_force_max = force_event.params["force_range"][1]
    
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * performance_thresh:
            new_force_max = min(current_force_max + increment, max_level)
            
            if new_force_max > current_force_max:
                force_event.params["force_range"] = (-new_force_max, new_force_max)
                force_event.params["torque_range"] = (-new_force_max, new_force_max)

    return torch.tensor(current_force_max, device=env.device)

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


def push_vel_event_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "orientation_exp",
    increment: float = 0.1,
    max_speed: float = 1.0,
    performance_thresh: float = 0.97,
) -> torch.Tensor:

    push_event = env.event_manager.get_term_cfg("push_robot")
    current_ranges = push_event.params["velocity_range"]
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * performance_thresh:
            current_max = current_ranges['x'][1]
            new_max = min(current_max + increment, max_speed)
            
            if new_max > current_max:
                push_event.params["velocity_range"] = {
                    "x": (-new_max, new_max),
                    "y": (-new_max, new_max),
                }
                current_ranges = push_event.params["velocity_range"]

    return torch.tensor(current_ranges['x'][1], device=env.device)
