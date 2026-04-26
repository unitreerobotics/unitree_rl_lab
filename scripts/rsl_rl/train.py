# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the game.")
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Training iterations to execute. Overrides the default value."
)
# append RSL-RL specific arguments
cli_args.add_rsl_rl_args(parser)
# parse arguments
args = parser.parse_args()
# append RSL-RL cli args
argcomplete.autocomplete(parser)


# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# reset the sys.path to avoid conflicts with the simulator
sys.path.pop(0)


# import after launching the simulator to avoid conflicts with the simulator
import gymnasium as gym
import numpy as np
import torch

from isaaclab_rl.rsl_rl.runners import OnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

# Import the configuration after the simulator is launched
if args.task is not None:
    # check if the task name is provided
    importlib.import_module(f"unitree_rl_lab.tasks.{args.task}")
else:
    # otherwise import all tasks
    import unitree_rl_lab.tasks


def main():
    """Main function."""
    # parse configuration
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg: OnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args=args)

    # create runner from configuration
    env = ManagerBasedRLEnv(cfg=env_cfg)
    runner = OnPolicyRunner(env, agent_cfg)

    # set seed for reproducibility
    if args.seed is not None:
        runner.set_seed(args.seed)

    # specify directory for logging experiments
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.abspath(log_dir)
    runner.logger.log_dir = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # write the video every N iterations
    if args.video:
        runner.logger.add_video(f"train_policy", 1)

    # set max iterations from command line arguments
    if args.max_iterations is not None:
        runner.learn(cfg=agent_cfg, max_iterations=args.max_iterations)
    else:
        runner.learn(cfg=agent_cfg)

    # close the simulator
    simulation_app.close()


if __name__ == "__main__":
    main()