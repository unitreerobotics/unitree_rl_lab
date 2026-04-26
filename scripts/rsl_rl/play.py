# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import gymnasium as gym
import numpy as np
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import argparse

import gymnasium as gym
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.parse_cfg import get_checkpoint_path
from omni.isaac.lab.utils.path import retrieve_checkpoint_path

from isaaclab_rl.rsl_rl.runners import OnPolicyRunner

# Import configuration
import cli_args  # isort: skip


def main():
    """Play with RSL-RL agent."""
    # parse arguments
    parser = argparse.ArgumentParser(description="Play policy with RSL-RL agent.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the game.")
    # append RSL-RL specific arguments
    cli_args.add_rsl_rl_args(parser)
    # parse arguments
    args = parser.parse_args()

    # import after launching the simulator to avoid conflicts with the simulator
    if args.task is not None:
        # check if the task name is provided
        importlib.import_module(f"unitree_rl_lab.tasks.{args.task}")
    else:
        # otherwise import all tasks
        import unitree_rl_lab.tasks

    # parse configuration
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg: cli_args.RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args=args)

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # create runner from configuration
    runner = OnPolicyRunner(env, agent_cfg, device=env.device)
    # load the trained policy
    # retrieve checkpoint path
    if agent_cfg.load_run:
        if agent_cfg.load_checkpoint is None:
            checkpoint_path = get_checkpoint_path(f"{agent_cfg.load_run}", "rsl_rl")
        else:
            checkpoint_path = retrieve_checkpoint_path(f"{agent_cfg.load_run}/{agent_cfg.load_checkpoint}")
        # load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        runner.load(checkpoint_path)
    else:
        raise ValueError("No checkpoint provided.")
    # switch to evaluation mode (turn off dropout for example)
    runner.policy.eval()

    # specify directory for logging experiments
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.abspath(log_dir)
    runner.logger.log_dir = log_dir

    # write the video every N iterations
    if args.video:
        video_index = 0
        video_writer = None
        video_frames = 0
        video_max_frames = args.video_length

    # set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # play with the trained policy
    count = 0
    obs, _ = env.get_observations()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # observe current state
            obs = obs.to(env.device)
            # compute actions
            actions = runner.policy(obs)[0]
            # apply actions
            obs, _, _, _ = env.step(actions)
            # increment counter
            count += 1
            # write video
            if args.video and count % args.video_interval == 0:
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        f"{log_dir}/videos/video_{video_index}.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        1 / env.physics_dt,
                        (env.viewport_camera.render_product.width, env.viewport_camera.render_product.height),
                    )
                # write current frame
                current_frame = env.viewport_camera.get_rgb()
                video_writer.write(current_frame)
                video_frames += 1
                # close video if max frames reached
                if video_frames >= video_max_frames:
                    video_writer.release()
                    video_writer = None
                    video_frames = 0
                    video_index += 1

    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()