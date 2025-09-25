import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.balance.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-23dof-StaggeredStance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.staggered_stance_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.staggered_stance_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.balance.agents.rsl_rl_ppo_cfg:BasePPORunnerStaggeredStanceCfg",
    },
)
