"""Check Gangnam Style reference geometry and randomized spawns in Isaac Lab 3.

Run with the Isaac Lab interpreter and --usd_path pointing to the G1 USD.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd_path", required=True)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

task = "Unitree-G1-29dof-Mimic-Gangnanm-Style"
cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=32)
cfg.scene.robot.spawn.usd_path = args.usd_path
cfg.commands.motion.debug_vis = False
cfg.scene.contact_forces.debug_vis = False
env = gym.make(task, cfg=cfg)

try:
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("motion")
    pelvis = robot.body_names.index("pelvis")
    torso = robot.body_names.index("torso_link")
    ankles = [robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    reference = command.motion._body_pos_w
    assert torch.all(reference[:, torso, 2] > reference[:, pelvis, 2]), "Reference torso is below pelvis"
    assert torch.all(reference[:, ankles, 2] < reference[:, pelvis, 2, None]), "Reference feet are above pelvis"
    for _ in range(8):
        env.reset()
        positions = robot.data.body_pos_w.torch
        up = torch.zeros((32, 3), device=env.unwrapped.device)
        up[:, 2] = 1.0
        upright = quat_apply(robot.data.root_quat_w.torch, up)[:, 2]
        assert torch.all(upright > 0.8), f"Inverted spawn: min upright={upright.min().item()}"
        assert torch.all(positions[:, torso, 2] > positions[:, pelvis, 2]), "Spawned torso is below pelvis"
        assert torch.all(positions[:, ankles, 2] < positions[:, pelvis, 2, None]), "Spawned feet are above pelvis"
    print(f"PASS: {len(reference)} reference frames and 256 randomized upright spawns", flush=True)
finally:
    env.close()
    app.close()
