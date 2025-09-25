import sys, os
from constants import JOINT_LIMITS
from utils import *

from matplotlib import pyplot as plt
import torch
import numpy as np
from lyapunov_trainer.loss import LyapunovRisk, CircleTuningLoss
from lyapunov_trainer.falsifier import SamplingBasedFalsifier, PGDFalsifier
from lyapunov_trainer.trainer import Trainer
from lyapunov_trainer.utils import Plot3D
from plot_lyapunov_model import plot_g1_balance_lyapunov_function
import numpy as np

from dataset import TrajectoryDataset
import gymnasium as gym 

class G1BalanceLyapunovTrainer(Trainer):
    def __init__(self, env, policy, lr, loss_fn, dt=0.02, n_inputs=46, hidden_sizes=None, circle_tuning_loss_fn=None, falsifier=None, device=None, lyapunov_by_construction=True, alpha=0.0):
        super().__init__(policy, lr, loss_fn, dt, n_inputs, hidden_sizes, circle_tuning_loss_fn, falsifier, device, lyapunov_by_construction, alpha)

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
        X: TensorDict (batch, num_states)
        '''
        obs = X['policy']
        return obs
        # old code: grabs subset of state since it is a stacked history.
        # base_ang_vel = X[:, 12:15]
        # joint_pos_start = 45 + 4*23
        # joint_pos_end = joint_pos_start + 23
        # joint_pos_rel = X[:, joint_pos_start:joint_pos_end]

        # joint_vel_start = joint_pos_end + 4*23
        # joint_vel_end = joint_vel_start + 23

        # joint_vel_rel = X[:, joint_vel_start:joint_vel_end]
        # if include_base_angle:
        #     filtered_x = torch.cat((base_ang_vel, joint_pos_rel, joint_vel_rel), dim=1)
        # else:
        #     filtered_x = torch.cat((joint_pos_rel, joint_vel_rel), dim=1)

        # return filtered_x
    
    
def plot_loss(true_loss, filename):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(len(true_loss)), true_loss, label='True Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)

def get_bounds_from_env(env, history=5):
    '''
    get the observation bounds for the policy
    '''
    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cpu"

    # projected_gravity
    pg_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    pg_max = torch.tensor([ 1.0,  1.0,  1.0], device=device)

    # base angular velocity
    bav_scale = 0.2
    # assume upper bound on angular velocity is 20 rad/s
    bav_lim = 20.0 * bav_scale
    bav_min = torch.full((3,), -bav_lim, device=device)
    bav_max = torch.full((3,),  bav_lim, device=device)

    # 3) velocity_commands (3) from task cfg
    cfg = getattr(env.unwrapped, 'cfg', None)
    vx_min, vx_max = cfg.commands.base_velocity.ranges.lin_vel_x
    vy_min, vy_max = cfg.commands.base_velocity.ranges.lin_vel_y
    wz_min, wz_max = cfg.commands.base_velocity.ranges.ang_vel_z
    cmd_min = torch.tensor([vx_min, vy_min, wz_min], device=device, dtype=torch.float32)
    cmd_max = torch.tensor([vx_max, vy_max, wz_max], device=device, dtype=torch.float32)
    # joint position and velocity
    
    # position limits (indices 0-22) from JOINT_LIMITS
    q_min_values = [JOINT_LIMITS[i][0] for i in range(23)] 
    q_max_values = [JOINT_LIMITS[i][1] for i in range(23)]
    q_min = torch.tensor(q_min_values, device=device)
    q_max = torch.tensor(q_max_values, device=device)
    
    # offset relative to reference position
    q_ref = env.unwrapped.scene["robot"].data.default_joint_pos[0]
    q_pos_min = (q_min - q_ref).to(device)
    q_pos_max = (q_max - q_ref).to(device)

    q_vel_scale = 0.05  # from obs_cfg.joint_vel_rel.scale
    q_vel_raw_min_values = [JOINT_LIMITS[i][0] for i in range(23, 46)]
    q_vel_raw_max_values = [JOINT_LIMITS[i][1] for i in range(23, 46)]
    q_vel_raw_min = torch.tensor(q_vel_raw_min_values, device=device)
    q_vel_raw_max = torch.tensor(q_vel_raw_max_values, device=device)
    q_vel_min = (q_vel_raw_min * q_vel_scale).to(device)
    q_vel_max = (q_vel_raw_max * q_vel_scale).to(device)

    # actions
    act_scale = 0.25
    act_min = q_pos_min * act_scale
    act_max = q_pos_max * act_scale

    # repeating to match observation history
    bav_min_hist = bav_min.repeat(history)
    bav_max_hist = bav_max.repeat(history)

    pg_min_hist = pg_min.repeat(history)   
    pg_max_hist = pg_max.repeat(history)
    
    cmd_min_hist = cmd_min.repeat(history) 
    cmd_max_hist = cmd_max.repeat(history)
    
    q_pos_min_hist = q_pos_min.repeat(history) 
    q_pos_max_hist = q_pos_max.repeat(history)
    
    q_vel_min_hist = q_vel_min.repeat(history) 
    q_vel_max_hist = q_vel_max.repeat(history)
    
    act_min_hist = act_min.repeat(history)
    act_max_hist = act_max.repeat(history)

    lb = torch.cat([bav_min_hist, pg_min_hist, cmd_min_hist, q_pos_min_hist, q_vel_min_hist, act_min_hist], dim=0)  # (390,)
    ub = torch.cat([bav_max_hist, pg_max_hist, cmd_max_hist, q_pos_max_hist, q_vel_max_hist, act_max_hist], dim=0)  # (390,)
    
    return lb, ub

def main():
    set_seed(42)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    env_id = 'Unitree-G1-23dof-Balance'
    policy_path = '/home/mht/research/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_balance/2025-09-23_10-21-13/model_3300.pt'
    print(f"Loading environment: {env_id}")
    print(f"Loading policy from: {policy_path}")
    env, policy = load_env_and_policy(env_id=env_id, policy_path=policy_path, episode_length_s=1.0)
    
    args = get_args()

    load_path = os.path.join(cur_dir, 'datasets', 'g1_balance_8_newton_500_traj_7.5s.npz')
    dataset = TrajectoryDataset(filename=load_path)

    history_len=5
    n_inputs = 390 # or 46 
    lie_offset = 0.01
    X_0 = torch.zeros(size=(n_inputs,))

    if n_inputs == 390:
        joint_pos_start = 45 + 4*23
        joint_pos_end = joint_pos_start + 23
        joint_vel_start = joint_pos_end + 4*23
        joint_vel_end = joint_vel_start + 23
        # equilibrium value for projected gravity
        projected_gravity = torch.tensor([0., 0., -1]*history_len, dtype=torch.float32)
        X_0[15:30] = projected_gravity
    else:
        joint_pos_start = 0
        joint_pos_end = joint_pos_start + 23
        joint_vel_start = joint_pos_end 
        joint_vel_end = joint_vel_start + 23
    # falsifier
    
    if args.falsifier:
        
        lb, ub = get_bounds_from_env(env, history=history_len)
        buffer_max_size = int(len(dataset)* (1.0 - args.test_split) * args.pct_counterexamples)
        # scale joint position, velocity,
        scale = torch.zeros(n_inputs, device=env.device)
        scale_factor = 0.05
        scale[joint_pos_start:joint_pos_end] = scale_factor
        scale[joint_vel_start:joint_vel_end] = scale_factor
        falsifier = SamplingBasedFalsifier(env, policy, lower_bound=lb, upper_bound=ub, epsilon=0, scale=scale, num_samples=1, 
                                           buffer_size=buffer_max_size, lyapunov_by_construction=args.lyapunov_by_construction, frequency=20, device=env.device)
    else:
        falsifier = None
    
    ### Start training process ###
    loss_fn = LyapunovRisk(lyapunov_factor=1.0, lie_factor=1.0, equilibrium_factor=1.0, lie_offset=lie_offset)
    # TODO remove or update with state min and max
    circle_tuning_loss_fn = None # CircleTuningLoss(state_max=torch.mean(state_max), tuning_factor=args.alpha)
    
    # Generate model name based on args
    hidden_str = '_'.join(map(str, args.hidden_sizes))
    model_name = 'lyapunov'
    
    if args.lyapunov_by_construction:
        model_name += '_by_construction'
    if args.falsifier:
        model_name += '_falsifier'

    model_name += f'_n_input_{n_inputs}_{hidden_str}_lr_{args.lr}_bs_{args.batch_size}_ep_{args.epochs}_lie_offset_{lie_offset}'

    save_path = os.path.join(cur_dir, 'logs', f'{model_name}.pt')

    trainer = G1BalanceLyapunovTrainer(env, policy, args.lr, loss_fn, dt=0.02, n_inputs=n_inputs, hidden_sizes=args.hidden_sizes,
                                       circle_tuning_loss_fn=circle_tuning_loss_fn, falsifier=falsifier, device=env.device, lyapunov_by_construction=args.lyapunov_by_construction, alpha=args.alpha)
    loss = trainer.train(dataset, X_0, epochs=args.epochs, verbose=True, batch_size=args.batch_size, test_split=args.test_split, decay_rate=1, run_name=model_name, pct_counterexamples=args.pct_counterexamples)
    # save model
    trainer.save_model(save_path)
    plot_loss(loss, os.path.join(cur_dir, 'results', f'{model_name}_loss.png'))

    model_load_path = save_path
    png_path = os.path.join(cur_dir, 'results', f'{model_name}.png')
    

    ### PLOTTING
    joint_pos_start = 45 + 4*23 if n_inputs == 390 else 0
    joint_pos_end = joint_pos_start + 23
    joint_vel_start = joint_pos_end + 4*23 if n_inputs == 390 else joint_pos_end
    idx1, idx2 = joint_pos_start, joint_vel_start  # left_hip_pitch position vs velocity
    plot_g1_balance_lyapunov_function(trainer, X_0, model_load_path, png_path, idx1=idx1, idx2=idx2)
    # close env and simulator
    env.close()
    close_down()

if __name__ == '__main__':
    main()
