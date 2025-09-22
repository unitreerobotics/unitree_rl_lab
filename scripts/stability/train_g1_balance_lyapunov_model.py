import sys, os
import argparse

from matplotlib import pyplot as plt
import torch
import numpy as np
from lyapunov_trainer.loss import LyapunovRisk, CircleTuningLoss
from lyapunov_trainer.falsifier import Falsifier
from lyapunov_trainer.trainer import Trainer
from lyapunov_trainer.utils import Plot3D
import numpy as np

from utils import *
from dataset import TrajectoryDataset
import gymnasium as gym 

class G1BalanceLyapunovTrainer(Trainer):
    def __init__(self, env, policy, lr, loss_fn, dt=0.02, n_inputs=46, hidden_sizes=None, circle_tuning_loss_fn=None, falsifier=None, device=None, lyapunov_by_construction=True, alpha=0.0):
        super().__init__(policy, lr, loss_fn, dt, n_inputs, hidden_sizes, circle_tuning_loss_fn, falsifier, device, lyapunov_by_construction, alpha)
        # use env to get s, a, s' pairs for falsifier and use finite difference approximation
        self.env = env

    def get_joint_pos(self, X):
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_pos_rel = X[:, joint_pos_start:joint_pos_end]
        return joint_pos_rel

    def get_joint_vel(self, X):
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_vel_start = joint_pos_end + 4*23
        joint_vel_end = joint_vel_start + 23
        joint_vel_rel = X[:, joint_vel_start:joint_vel_end]
        return joint_vel_rel

    def get_base_ang_vel(self, X):
        return X[:, 12:15]

    def process_state(self, X, include_base_angle=False):
        '''
        Custom function for input to lyapunov model.
        X: (batch, num_states)
        '''
        return X
        base_ang_vel = X[:, 12:15]
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_pos_rel = X[:, joint_pos_start:joint_pos_end]

        joint_vel_start = joint_pos_end + 4*23
        joint_vel_end = joint_vel_start + 23

        joint_vel_rel = X[:, joint_vel_start:joint_vel_end]
        if include_base_angle:
            filtered_x = torch.cat((base_ang_vel, joint_pos_rel, joint_vel_rel), dim=1)
        else:
            filtered_x = torch.cat((joint_pos_rel, joint_vel_rel), dim=1)

        return filtered_x
    
    def step(self, X, u):
        '''
        Generates all X_primes needed given current state and current action
        X: state
        u: action
        '''
        N = X.shape[0]
        X_prime = torch.empty_like(X)
        for i in range(N):
            x_i = X[i, :].unsqueeze(0)
            
            current_sim_state = self.env.unwrapped.scene.get_state()
            current_robot_state = self.get_robot_state(x_i)
            current_sim_state['articulation']['robot'] = current_robot_state
            self.env.unwrapped.reset_to(current_sim_state, env_ids=None)
            # get current action to take 
            u_i = u[i, :].unsqueeze(0)
            # take step in environment
            obs, reward, terminated, truncated, info = self.env.step(u_i)
            # get next state
            X_prime[i, :] = obs.squeeze(0)

        return X_prime
    
    def get_robot_state(self, obs):
        # relative joint pos and vel
        joint_pos_rel = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)

        default_joint_pos = self.env.unwrapped.scene["robot"].data.default_joint_pos[0]
        joint_pos = joint_pos_rel + default_joint_pos

        root_velocity = torch.zeros(1, 6)
        root_velocity[:, 3:6] = self.get_base_ang_vel(obs)
        # initial pose for unitree g1
        root_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        robot_state = {
            'root_pose': root_pose,
            'root_velocity': root_velocity,
            'joint_position': joint_pos.unsqueeze(0),
            'joint_velocity': joint_vel.unsqueeze(0)
        }
        return robot_state
    
def plot_loss(true_loss, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(len(true_loss)), true_loss, label='True Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig(filename)

def main():
    set_seed(42)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    env_id = 'Unitree-G1-23dof-Balance'
    policy_path = os.path.join(cur_dir, '/hdd/users/mat028/research/unitree_rl_lab/scripts/stability/logs/2025-08-27_16-50-30/model_1100.pt')
    print(f"Loading environment: {env_id}")
    print(f"Loading policy from: {policy_path}")
    env, policy = load_env_and_policy(env_id=env_id, policy_path=policy_path, episode_length_s=1.0)
    
    args = get_args()


    # state, action, next_state dataset
    load_path = os.path.join(cur_dir, 'logs', 'g1_balance_5_newton_100_traj_1s.npz')
    dataset = TrajectoryDataset(filename=load_path, state_action_only=False)

    # for data 

    # # TODO state bounds for falsifier
    # state_min = torch.min(dataset.states, dim=0).values
    # state_max = torch.max(dataset.states, dim=0).values

    # # TODO if working with full state space, not all equilibrium positions are 0. (e.g. gravity term)
    # n_inputs = 390 #49  # joint positions + joint velocities
    # X_0 = torch.zeros(size=(n_inputs,))

    # # falsifier_
    # if args.falsifier:
    #     falsifier = Falsifier(state_min, state_max, epsilon=0., scale=0.05, frequency=100, num_samples=5)
    # else:
    #     falsifier = None
    
    # ### Start training process ###
    # loss_fn = LyapunovRisk(lyapunov_factor=1.0, lie_factor=1.0, equilibrium_factor=1.0, lie_offset=1e-3)
    # circle_tuning_loss_fn = None # CircleTuningLoss(state_max=torch.mean(state_max), tuning_factor=args.alpha)
    

    # save_path = os.path.join(cur_dir, 'logs', 'lyapunov_model.pt')

    # trainer = G1BalanceLyapunovTrainer(env, policy, args.lr, loss_fn, dt=0.02, n_inputs=n_inputs, hidden_sizes=args.hidden_sizes,
    #                                    circle_tuning_loss_fn=circle_tuning_loss_fn, falsifier=falsifier, device=env.device, lyapunov_by_construction=args.lyapunov_by_construction, alpha=args.alpha)
    # loss = trainer.train(dataset, X_0, epochs=args.epochs, verbose=True, batch_size=args.batch_size, test_split=args.test_split, decay_rate=1)
    # # save model
    # trainer.save_model(save_path)
    # plot_loss(loss, os.path.join(cur_dir, 'logs', 'lyapunov_loss.png'))
        
    # close env and simulator
    env.close()
    close_down()

if __name__ == '__main__':
    main()
