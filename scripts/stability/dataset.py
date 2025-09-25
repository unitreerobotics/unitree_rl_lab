
from utils import *
from tensordict import TensorDict

import numpy as np
import sys, os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import gymnasium as gym

'''
Trajectory Collection for the IsaacLab G1 Balance Environment.
History size of 5 is used (original observation is size 78)
+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (390,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_ang_vel                   |    (15,)    |
|     1     | projected_gravity              |    (15,)    |
|     2     | velocity_commands              |    (15,)    |
|     3     | joint_pos_rel                  |    (115,)   |
|     4     | joint_vel_rel                  |    (115,)   |
|     5     | last_action                    |    (115,)   |
+-----------+--------------------------------+-------------+
+----------------------------------------------------------+
| Active Observation Terms in Group: 'critic' (shape: (405,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_lin_vel                   |    (15,)    |
|     1     | base_ang_vel                   |    (15,)    |
|     2     | projected_gravity              |    (15,)    |
|     3     | velocity_commands              |    (15,)    |
|     4     | joint_pos_rel                  |    (115,)   |
|     5     | joint_vel_rel                  |    (115,)   |
|     6     | last_action                    |    (115,)   |
+-----------+--------------------------------+-------------+
[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+-----------------------------------------+
|     Active Action Terms (shape: 23)     |
+-------+---------------------+-----------+
| Index | Name                | Dimension |
+-------+---------------------+-----------+
|   0   | JointPositionAction |        23 | 
+-------+---------------------+-----------+
TERMINATIONS
sim dt = 0.005
control dt = 0.02
episode length: 20 seconds
max episode length (timesteps): 1000
'''

def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_cpu(v) for v in x]
        return type(x)(t) if not isinstance(x, list) else t
    return x

class TrajectoryCollector():

    def __init__(self, env, policy, N=500):
        self.env = env
        # numer of different trajectories
        self.N = N
        # trained torch policy
        self.policy = policy

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        with torch.inference_mode():
            action = self.policy(x)
        return action

    def build(self):
        trajectories = []
        for i in tqdm(range(self.N)):
            # reset environment for new trajectory
            obs = self.env.get_observations()
            # list of tuples of (s, pi(s), s') pairs
            trajectory = self.get_trajectory(obs)
            trajectories.append(trajectory)

        return trajectories

    def get_trajectory(self, x):
        terminate = False

        trajectory = []
        while not terminate:
            # get action for current state x
            try:
                sim_state = self.env.unwrapped.scene.get_state()
            except Exception:
                raise ValueError("IsaacLab environment missing scene.get_state.")
            a = self.get_action(x)
            # get next state
            x_prime, reward, terminate, info = self.env.step(a)

            if ('policy' in x) and ('critic' in x):
                state_policy = x['policy'][0]
                state_critic = x['critic'][0]
                next_state_policy = x_prime['policy'][0]
                next_state_critic = x_prime['critic'][0]
                action = a[0]

                trajectory.append((state_policy, state_critic, action, next_state_policy, next_state_critic, sim_state))
            else:
                raise ValueError("Expected TensorDict with 'policy' and 'critic' keys.")

            # update x to be next state
            x = x_prime

        return trajectory

    def save(self, trajectories, filename='trajectories.npz'):
        flattened_data = self.flatten_trajectories(trajectories)
        # Ensure directory exists
        dirpath = os.path.dirname(filename)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        np.savez_compressed(filename, **flattened_data)

    def flatten_trajectories(self, trajectories):
        episode_lengths = []

        obs_policy_list = []
        obs_critic_list = []
        actions_list = []
        next_obs_policy_list = []
        next_obs_critic_list = []
        sim_state_list = []

        for trajectory in trajectories:
            episode_lengths.append(len(trajectory))
            for data in trajectory:
                state_policy, state_critic, action, next_state_policy, next_state_critic, sim_state = data
                obs_policy_list.append(state_policy.detach().cpu().numpy())
                obs_critic_list.append(state_critic.detach().cpu().numpy())
                actions_list.append(action.detach().cpu().numpy())
                next_obs_policy_list.append(next_state_policy.detach().cpu().numpy())
                next_obs_critic_list.append(next_state_critic.detach().cpu().numpy())
                sim_state_list.append(_to_cpu(sim_state))

        data_dict = {
            'obs_policy': np.asarray(obs_policy_list),
            'obs_critic': np.asarray(obs_critic_list),
            'actions': np.asarray(actions_list),
            'next_obs_policy': np.asarray(next_obs_policy_list),
            'next_obs_critic': np.asarray(next_obs_critic_list),
            'episode_lengths': np.asarray(episode_lengths, dtype=np.int32),
            'sim_state': np.asarray(sim_state_list, dtype=object),
        }
        return data_dict

class TrajectoryDataset(Dataset):
    def __init__(self, filename):
        # dataset of flattened tuples with dict-style observations
        loaded = self.load_dataset(filename)
        self.episode_lengths = torch.tensor(loaded['episode_lengths'], dtype=torch.int32)

        self.states_policy = torch.tensor(loaded['obs_policy'], dtype=torch.float32)
        self.states_critic = torch.tensor(loaded['obs_critic'], dtype=torch.float32)
        self.actions = torch.tensor(loaded['actions'], dtype=torch.float32)
        self.next_states_policy = torch.tensor(loaded['next_obs_policy'], dtype=torch.float32)
        self.next_states_critic = torch.tensor(loaded['next_obs_critic'], dtype=torch.float32)
        self.sim_state = loaded['sim_state']

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

def collate_tensordict(batch):
    obs_list, actions_list, next_obs_list, sim_states = zip(*batch)
    obs_policy = torch.stack([b['policy'] for b in obs_list], dim=0)
    obs_critic = torch.stack([b['critic'] for b in obs_list], dim=0)
    next_policy = torch.stack([b['policy'] for b in next_obs_list], dim=0)
    next_critic = torch.stack([b['critic'] for b in next_obs_list], dim=0)
    actions = torch.stack(actions_list, dim=0)

    obs_td = TensorDict({'policy': obs_policy, 'critic': obs_critic}, batch_size=[obs_policy.shape[0]])
    next_td = TensorDict({'policy': next_policy, 'critic': next_critic}, batch_size=[next_policy.shape[0]])
    # sim_states is a tuple of dicts; return as list to keep objects intact
    return obs_td, actions, next_td, list(sim_states)



if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # save path
    num_trajectories = 500
    save_path = os.path.join(cur_dir, 'datasets', 'g1_balance_8_newton_{}_traj_10s.npz'.format(num_trajectories))

    env_id = 'Unitree-G1-23dof-Balance'
    policy_path = '/home/mht/research/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_balance/2025-09-23_10-21-13/model_3300.pt'
    print(f"Loading environment: {env_id}")
    print(f"Loading policy from: {policy_path}")
    
    env, policy = load_env_and_policy(env_id=env_id, policy_path=policy_path, episode_length_s=7.5)

    print('##### Data Collection ######')

    dataset = TrajectoryCollector(env, policy=policy, N=num_trajectories)
    print('Building Dataset...')
    trajectories = dataset.build()


    print('Saving flattened dataset of {} tracectories...'.format(len(trajectories)))
    dataset.save(trajectories, filename=save_path)

    print('##### Pytorch Dataset ######')
    load_path = save_path
    loaded_data = TrajectoryDataset(load_path)
    print('Loaded {} transitions.'.format(len(loaded_data)))

    dataloader = DataLoader(
        loaded_data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tensordict 
    )

    obs, actions, next_obs, sim_states = next(iter(dataloader))
    print(f"Obs policy shape: {obs['policy'].shape}")
    print(f"Obs critic shape: {obs['critic'].shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next obs policy shape: {next_obs['policy'].shape}")
    print(f"Next obs critic shape: {next_obs['critic'].shape}")
    print(f"Sim states batch size: {len(sim_states)}; type of one entry: {type(sim_states[0])}")
    

    # close env and simulator
    env.close()
    close_down()