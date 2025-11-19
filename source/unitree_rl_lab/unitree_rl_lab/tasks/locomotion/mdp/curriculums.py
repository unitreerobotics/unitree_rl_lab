from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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



def terrain_levels(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    up_ratio: float = 0.5, 
    down_ratio: float = 0.5,             
) -> torch.Tensor:
    """
    A robust curriculum for terrain difficulty that is only called at the end of an episode.

    - **Upgrade condition**: Based on the total displacement and the size of the terrain.
    - **Downgrade condition**: Based on the expected distance covered during the last command window.
      This prevents agents that are slow in the beginning but fast at the end from being unfairly
      downgraded.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # Total episode duration in seconds
    T_episode = float(getattr(env, "max_episode_length_s", 0.0))
    
    # Magnitude of the last linear velocity command
    command_term = env.command_manager.get_term("base_velocity")
    command = env.command_manager.get_command("base_velocity")
    cmd_speed_last = torch.linalg.norm(command[env_ids, :2], dim=1)
    cmd_sampling_period_s = command_term.cfg.resampling_time_range[0]

    terrain_size_x = float(terrain.cfg.terrain_generator.size[0])
    move_up = distance > (up_ratio * terrain_size_x)

    # Conservatively estimate the expected distance using only the last command window
    effective_T = min(T_episode, float(cmd_sampling_period_s))
    expected_dist_last = cmd_speed_last * effective_T

    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < (down_ratio * expected_dist_last)
    move_down *= ~move_up

    # Do not downgrade if the environment is reset due to a timeout
    move_down[env.termination_manager.time_outs[env_ids]] = False

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
