# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file name to resume from.")


def parse_rsl_rl_cfg(task_name: str, args: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration file for RSL-RL agent.

    Args:
        task_name: Name of the task.
        args: Command line arguments.

    Returns:
        The configuration class for RSL-RL agent.
    """
    # import configuration
    if task_name.startswith("Unitree"):
        from unitree_rl_lab.tasks import unitree_a1_cfgs

        # check if the task name is provided
        if task_name == "UnitreeA1TerrainEnv-v0":
            cfg = unitree_a1_cfgs.UnitreeA1RoughCfg()
            cfg_class = unitree_a1_cfgs.UnitreeA1RoughCfgPPO
        else:
            raise ValueError(f"Task {task_name} not found!")
    else:
        raise ValueError(f"Task {task_name} not supported!")

    # update runner configuration with command line arguments
    cfg_class.experiment_name = args.experiment_name
    cfg_class.run_name = args.run_name
    cfg_class.resume = args.resume
    cfg_class.load_run = args.load_run
    cfg_class.load_checkpoint = args.checkpoint

    # create runner configuration
    runner_cfg = cfg_class()
    runner_cfg.policy = cfg

    # set maximum iterations if provided
    if args.max_iterations is not None:
        runner_cfg.max_iterations = args.max_iterations

    return runner_cfg