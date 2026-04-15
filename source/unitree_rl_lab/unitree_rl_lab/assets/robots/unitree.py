# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots import unitree_actuators

UNITREE_MODEL_DIR = "path/to/unitree_model"  # Replace with the actual path to your unitree_model directory
UNITREE_ROS_DIR = "path/to/unitree_ros"  # Replace with the actual path to your unitree_ros package


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


""" Configuration for the Unitree robots."""

UNITREE_GO2_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)

UNITREE_GO2W_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2w_description/urdf/go2w_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_.*": 25.0,
                ".*_thigh_.*": 25.0,
                ".*_calf_.*": 25.0,
                ".*_foot_.*": 0,
            },
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
    ],
    # fmt: on
)

UNITREE_B2_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/b2_description/urdf/b2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/B2/usd/b2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
    },
    joint_sdk_names=UNITREE_GO2_CFG.joint_sdk_names.copy(),
)

UNITREE_H1_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/h1_description/urdf/h1.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_elbow_joint": 0.32,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV-1": IdealPDActuatorCfg(
            joint_names_expr=[".*ankle.*", ".*_shoulder_pitch_.*", ".*_shoulder_roll_.*"],
            effort_limit=40,
            velocity_limit=9,
            stiffness={
                ".*ankle.*": 40.0,
                ".*_shoulder_.*": 100.0,
            },
            damping=2.0,
            armature=0.01,
        ),
        "GO2HV-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_.*", ".*_elbow_.*"],
            effort_limit=18,
            velocity_limit=20,
            stiffness=50,
            damping=2.0,
            armature=0.01,
        ),
        "M107-24-1": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=300.0,
            velocity_limit=14.0,
            stiffness=200.0,
            damping=4.0,
            armature=0.01,
        ),
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", "torso_joint"],
            effort_limit=200,
            velocity_limit=23.0,
            stiffness={
                ".*_hip_.*": 150.0,
                "torso_joint": 300.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "torso_joint": 6.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "torso_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "",
        "left_ankle_joint",
        "right_ankle_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ],
)

UNITREE_G1_23DOF_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_23dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],  # 5
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],  # 4
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll_.*"],  # 10
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],  # 4
            effort_limit_sim=35,
            velocity_limit_sim=30,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
)

UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # 14.25062309787429
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # 40.17923847137318
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # 99.09842777666113
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # 16.77832748089279

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # 0.907222843292423
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 2.5578897650279457
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 6.3088018534966395
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # 1.06814150219

UNITREE_G1_29DOF_MIMIC_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)

UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_lock_waist_rev_1_0.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)

UNITREE_G1_23DOF_MIMIC_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
)

UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {}
for a in UNITREE_G1_29DOF_MIMIC_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            UNITREE_G1_29DOF_MIMIC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_ACTION_SCALE = {}
for a in UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            UNITREE_G1_29DOF_LOCK_WAIST_MIMIC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

UNITREE_G1_23DOF_MIMIC_ACTION_SCALE = {}
for a in UNITREE_G1_23DOF_MIMIC_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            UNITREE_G1_23DOF_MIMIC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
