"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py -f path_to_motion.npz --robot_model g1_29dof
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--file", "-f", type=str, required=True)
parser.add_argument(
    "--robot_model",
    type=str,
    default="g1_29dof",
    choices=("g1_23dof", "g1_29dof", "g1_29dof_lock_waist"),
    help="The robot model used to replay the motion.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_MIMIC_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_CFG
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
from unitree_rl_lab.tasks.mimic.mdp import MotionLoader

ROBOT_CFG_MAP = {
    "g1_23dof": UNITREE_G1_23DOF_MIMIC_CFG,
    "g1_29dof": UNITREE_G1_29DOF_MIMIC_CFG,
    "g1_29dof_lock_waist": UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_CFG,
}
ROBOT_CFG = ROBOT_CFG_MAP[args_cli.robot_model]

##
# Pre-defined configs
##


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    robot_joints = robot.data.joint_pos.shape[1]
    motion_joints = motion.joint_pos.shape[1]
    if motion_joints != robot_joints:
        if motion_joints == 29 and robot_joints == 27:
            motion.joint_pos = torch.cat([motion.joint_pos[:, :13], motion.joint_pos[:, 15:]], dim=1)
            motion.joint_vel = torch.cat([motion.joint_vel[:, :13], motion.joint_vel[:, 15:]], dim=1)
        elif motion_joints == 29 and robot_joints == 23:
            motion.joint_pos = torch.cat(
                [motion.joint_pos[:, :13], motion.joint_pos[:, 15:20], motion.joint_pos[:, 22:27]], dim=1
            )
            motion.joint_vel = torch.cat(
                [motion.joint_vel[:, :13], motion.joint_vel[:, 15:20], motion.joint_vel[:, 22:27]], dim=1
            )
        else:
            raise ValueError(
                f"Motion joint dim ({motion_joints}) does not match robot joint dim ({robot_joints})."
            )

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
