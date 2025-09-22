import sys, os
import argparse

from matplotlib import pyplot as plt
import torch
import numpy as np
from lyapunov_trainer.loss import LyapunovRisk
from lyapunov_trainer.falsifier import Falsifier
from train_g1_balance_lyapunov_model import G1BalanceLyapunovTrainer
from lyapunov_trainer.utils import Plot3D
import numpy as np

from utils import *

def plot_lyapunov_3d(trainer, equilibrium_states, filename, x_min, x_max, y_min, y_max, n_inputs=46, idx1=0, idx2=23, device='cpu', label1='State 1', label2='State 2', factor_1=0.1, factor_2=0.1):
    """Simple 3D plot of the Lyapunov function for two specified state indices
    
    Args:
        model: Neural Lyapunov model
        filename: Output filename for the plot
        x_min, x_max, y_min, y_max: Required ranges for the plot axes
        n_inputs: Number of input dimensions
        idx1, idx2: Indices of the two state dimensions to plot
        device: Device to run the model on
        label1, label2: Labels for the x and y axes
        factor: Factor to scale the ranges (default 0.1 = multiply ranges by 0.1)
    """
    
    num_axis_points = 100 
    
    # Apply factor to scale the ranges (simple multiplication)
    x_min_scaled = x_min * factor_1
    x_max_scaled = x_max * factor_1
    y_min_scaled = y_min * factor_2
    y_max_scaled = y_max * factor_2
    
    # Create grid for the two specified state variables
    x = np.linspace(x_min_scaled, x_max_scaled, num_axis_points)
    y = np.linspace(y_min_scaled, y_max_scaled, num_axis_points)
    X, Y = np.meshgrid(x, y)
    
    # Create states (zeros except for the two specified indices)
    states = torch.zeros(num_axis_points**2, n_inputs)
    states[:, idx1] = torch.from_numpy(X.flatten()).float()
    states[:, idx2] = torch.from_numpy(Y.flatten()).float()
    
    with torch.no_grad():
        V_values = trainer.get_lyapunov_output(states.to(device), equilibrium_states.to(device)).cpu().numpy()
    V = V_values.reshape(num_axis_points, num_axis_points)
    
    Plot3D(X, Y, V, filename=filename, xlabel=label1, ylabel=label2)
    print("Plot saved successfully in `{}`.".format(filename))


def plot_g1_balance_lyapunov_function(trainer, load_path, n_inputs):
    x_0 = torch.zeros(size=(n_inputs,))
    offset = n_inputs - 46
    
    trainer.load_model(load_path)
    joint_names = [
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
        "right_wrist_roll_joint"
    ]
    pos_joint_names = [name + '_pos' for name in joint_names]
    vel_joint_names = [name + '_vel' for name in joint_names]
    joint_names = pos_joint_names + vel_joint_names
    joint_limits = {
        -3: None,
        -2: None,
        -1: None,
        0: (-1.54, 1.54),
        1: (-1.54, 1.54),
        2: (-1.54, 1.54),
        3: (-2.43, 2.43),
        4: (-0.87, 0.87),
        5: (-0.87, 0.87),
        6: (-1.54, 1.54),
        7: (-1.54, 1.54),
        8: (-1.54, 1.54),
        9: (-2.43, 2.43),
        10: (-0.87, 0.87),
        11: (-0.87, 0.87),
        12: (-1.54, 1.54),
        13: (-0.44, 0.44),
        14: (-0.44, 0.44),
        15: (-0.44, 0.44),
        16: (-0.44, 0.44), 
        17: (-0.44, 0.44),
        18: (-0.44, 0.44),
        19: (-0.44, 0.44),
        20: (-0.44, 0.44),
        21: (-0.44, 0.44),  
        22: (-0.44, 0.44),
        
        23: (-32, 32),     
        24: (-32, 32),    
        25: (-32, 32),   
        26: (-20, 20),
        27: (-30, 30),   
        28: (-30, 30),      
        29: (-32, 32),      
        30: (-32, 32),     
        31: (-32, 32),    
        32: (-20, 20), 
        33: (-30, 30),     
        34: (-30, 30),     
        35: (-32, 32),
        36: (-37, 37),      
        37: (-37, 37),     
        38: (-37, 37),   
        39: (-37, 37),
        40: (-37, 37),     
        41: (-37, 37),      
        42: (-37, 37),      
        43: (-37, 37),      
        44: (-37, 37), 
        45: (-37, 37),      
    }
    
    # Plot 3D Lyapunov function
    idx1, idx2 = 0, 23  # left_hip_pitch position vs velocity
    x_min, x_max = joint_limits[idx1]
    y_min, y_max = joint_limits[idx2]
    idx1 = idx1 + offset
    idx2 = idx2 + offset
    plot_lyapunov_3d(trainer, x_0, os.path.join(cur_dir, 'logs', 'lyapunov_3d.png'), 
                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                     n_inputs=n_inputs, idx1=idx1, idx2=idx2, device=trainer.device,
                     label1=f"{joint_names[idx1]}", label2=f"{joint_names[idx2]}", factor_1=0.2, factor_2=0.02)
    

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    env_id = 'Unitree-G1-23dof-Balance'
    policy_path = os.path.join(cur_dir, '/hdd/users/mat028/research/unitree_rl_lab/scripts/stability/logs/2025-08-27_16-50-30/model_1100.pt')
    print(f"Loading environment: {env_id}")
    print(f"Loading policy from: {policy_path}")
    env, policy = load_env_and_policy(env_id=env_id, policy_path=policy_path, episode_length_s=10.0)
    
    args = get_args()


    ### Start training process ###
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1., lie_offset=0.1)
    

    load_path = os.path.join(cur_dir, 'logs', 'lyapunov_model.pt')


    n_inputs = 49
    trainer = G1BalanceLyapunovTrainer(env, policy, args.lr, loss_fn, dt=0.02, n_inputs=n_inputs, hidden_sizes=args.hidden_sizes,
                                       device=env.device, lyapunov_by_construction=args.lyapunov_by_construction, alpha=args.alpha)
    

    plot_g1_balance_lyapunov_function(trainer, load_path, n_inputs)
    # close env and simulator
    env.close()
    close_down()