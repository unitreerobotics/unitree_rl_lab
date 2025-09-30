import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

class TrajectoryDataset(Dataset):
    def __init__(self, filename):
        loaded = self.load_dataset(filename)
        self.episode_lengths = torch.tensor(loaded['episode_lengths'], dtype=torch.int32)
        
        self.flattened_states_policy = torch.tensor(loaded['obs_policy'], dtype=torch.float32)
        self.flattened_states_critic = torch.tensor(loaded['obs_critic'], dtype=torch.float32)
        self.flattened_actions = torch.tensor(loaded['actions'], dtype=torch.float32)
        self.flattened_next_states_policy = torch.tensor(loaded['next_obs_policy'], dtype=torch.float32)
        self.flattened_next_states_critic = torch.tensor(loaded['next_obs_critic'], dtype=torch.float32)
        self.flattened_sim_state = loaded['sim_state']
        self._unflatten_rollouts()

    def _unflatten_rollouts(self):

        self.states_policy = []
        self.states_critic = []
        self.actions = []
        self.next_states_policy = []
        self.next_states_critic = []
        self.sim_state = []

        start_step = 0
        end_step = 0
        for episode_length in self.episode_lengths:
            end_step += episode_length
            self.states_policy.append(self.flattened_states_policy[start_step:end_step])
            self.states_critic.append(self.flattened_states_critic[start_step:end_step])
            self.actions.append(self.flattened_actions[start_step:end_step])
            self.next_states_policy.append(self.flattened_next_states_policy[start_step:end_step])
            self.next_states_critic.append(self.flattened_next_states_critic[start_step:end_step])
            self.sim_state.append(list(self.flattened_sim_state[start_step:end_step]))
            start_step += episode_length

    def load_dataset(self, filename):
        '''
        Loads named arrays from .npz and returns a dict
        '''
        loaded = np.load(filename, allow_pickle=True)
        return {key: loaded[key] for key in loaded}
    
    def __len__(self):
        return len(self.states_policy)

    def __getitem__(self, idx):
        policy = self.states_policy[idx]
        critic = self.states_critic[idx]
        next_policy = self.next_states_policy[idx]
        next_critic = self.next_states_critic[idx]

        obs = {'policy': policy, 'critic': critic}
        next_obs = {'policy': next_policy, 'critic': next_critic}
        sim_state = self.sim_state[idx]
        return obs, self.actions[idx], next_obs, sim_state


def plot_trajectory_projections(obs, actions, next_obs, sim_states, axes, ylimit=True):
    default_joint_pos = torch.tensor([-0.1000, -0.1000,  0.0000,  0.0000,  0.0000,  0.3000,  0.3000,  0.0000,
                                       0.0000,  0.2500, -0.2500,  0.3000,  0.3000,  0.0000,  0.0000, -0.2000,
                                       -0.2000,  0.9700,  0.9700,  0.0000,  0.0000,  0.1500, -0.1500], dtype=torch.float32)
    # x, y pos can vary, but target height is 0.78 from training
    default_root_pos = torch.tensor([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    policy_obs = obs['policy'][0]
    critic_obs = obs['critic'][0]
    actions = actions[0]
    next_policy_obs = next_obs['policy'][0]
    next_critic_obs = next_obs['critic'][0]


    pos_norms = []
    vel_norms = []
    ang_vel_norms = []
    rpy_norms = []

    # Joint stability metrics
    right_hip_pos_norms = []
    right_hip_vel_norms = []
    left_knee_pos_norms = []
    left_knee_vel_norms = []
    
    for i in range(len(sim_states)):
        robot_state = sim_states[i]['articulation']['robot']
        root_pos = robot_state['root_pose'].reshape(-1)
        root_vel = robot_state['root_velocity'].reshape(-1)
        joint_pos = robot_state['joint_position'].reshape(-1)
        joint_vel = robot_state['joint_velocity'].reshape(-1)
        
        if i == 0:
            default_root_pos[0:2] = root_pos[0:2]
            default_rpy = Rotation.from_quat(root_pos[3:7], scalar_first=True).as_euler('xyz', degrees=False)

        lin_pos = root_pos.reshape(-1)[0:3] - default_root_pos[0:3]
        pos_norms.append(torch.linalg.norm(lin_pos).item())

        lin_vel = root_vel[0:3]
        vel_norms.append(torch.linalg.norm(lin_vel).item())

        ang_vel = root_vel[3:6]
        ang_vel_norms.append(torch.linalg.norm(ang_vel).item())
        
        quat = root_pos[3:7] 
        rpy_relative = Rotation.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False) - default_rpy
        rpy_norms.append(np.linalg.norm(rpy_relative))
        
        right_hip_roll = joint_pos[0] - default_joint_pos[0]
        right_hip_pitch = joint_pos[1] - default_joint_pos[1] 
        right_hip_yaw = joint_pos[2] - default_joint_pos[2]
        right_hip_pos_norms.append(np.linalg.norm([right_hip_roll, right_hip_pitch, right_hip_yaw]))
        
        # Right hip velocity norm
        right_hip_roll_vel = joint_vel[0]
        right_hip_pitch_vel = joint_vel[1]
        right_hip_yaw_vel = joint_vel[2]
        right_hip_vel_norms.append(np.linalg.norm([right_hip_roll_vel, right_hip_pitch_vel, right_hip_yaw_vel]))
        
        left_knee_pitch = joint_pos[9] - default_joint_pos[9]
        left_knee_pos_norms.append(abs(left_knee_pitch))
        
        # Left knee velocity
        left_knee_vel = joint_vel[9]
        left_knee_vel_norms.append(abs(left_knee_vel))

    axes[0, 0].plot(pos_norms)
    axes[0, 0].set_title('base position')
    axes[0, 0].set_ylabel('projected error')
    axes[0, 0].grid(True, alpha=0.3)


    axes[1, 0].plot(vel_norms)
    axes[1, 0].set_title('base velocity')
    axes[1, 0].set_xlabel('timestep')
    axes[1, 0].set_ylabel('projected error')
    axes[1, 0].grid(True, alpha=0.3)


    axes[0, 1].plot(rpy_norms)
    axes[0, 1].set_title('base angle')
    axes[0, 1].grid(True, alpha=0.3)


    axes[1, 1].plot(ang_vel_norms)
    axes[1, 1].set_title('base angular velocity')
    axes[1, 1].set_xlabel('timestep')
    axes[1, 1].grid(True, alpha=0.3)

    
    axes[0, 2].plot(right_hip_pos_norms)
    axes[0, 2].set_title('right hip position')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 0.4])
    
    axes[1, 2].plot(right_hip_vel_norms)
    axes[1, 2].set_title('right hip velocity')
    axes[1, 2].set_xlabel('timestep')
    axes[1, 2].grid(True, alpha=0.3)

    
    axes[0, 3].plot(left_knee_pos_norms)
    axes[0, 3].set_title('left knee position')
    axes[0, 3].grid(True, alpha=0.3)

    
    axes[1, 3].plot(left_knee_vel_norms)
    axes[1, 3].set_title('left knee velocity')
    axes[1, 3].set_xlabel('timestep')
    axes[1, 3].grid(True, alpha=0.3)

    if ylimit:
        axes[0, 0].set_ylim([0, 0.1])
        axes[1, 0].set_ylim([0, 1.5])
        axes[0, 1].set_ylim([0, 0.2])
        axes[1, 1].set_ylim([0, 1.5])
        axes[0, 2].set_ylim([0, 0.1])
        axes[1, 2].set_ylim([0, 1.5])
        axes[0, 3].set_ylim([0, 0.1])
        axes[1, 3].set_ylim([0, 1.5])


def visualize_trajectories(trajectories, num_trajectories=1, filename: str | None =None, ylimit=True):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    
    plt.subplots_adjust(hspace=0.2, wspace=0.3) 
    
    trajectory_count = 0
    for trajectory in trajectories:
        if trajectory_count == num_trajectories:
            break
        obs, actions, next_obs, sim_states = trajectory
        plot_trajectory_projections(obs, actions, next_obs, sim_states, axes, ylimit=ylimit)
        trajectory_count += 1

    # axes[0, 0].legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)