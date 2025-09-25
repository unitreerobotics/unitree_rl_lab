import sys, os
import argparse

from matplotlib import pyplot as plt
import torch
import numpy as np
from lyapunov_trainer.utils import Plot3D
import numpy as np

from utils import *
from constants import *
def plot_lyapunov_3d(trainer, equilibrium_states, filename, x_min, x_max, y_min, y_max, idx1=0, idx2=23, device='cpu', label1='State 1', label2='State 2', factor_1=0.1, factor_2=0.1):
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
    
    # Apply factor to scale the ranges
    x_min_scaled = x_min * factor_1
    x_max_scaled = x_max * factor_1
    y_min_scaled = y_min * factor_2
    y_max_scaled = y_max * factor_2
    
    # Create grid for the two specified state variables
    x = np.linspace(x_min_scaled, x_max_scaled, num_axis_points)
    y = np.linspace(y_min_scaled, y_max_scaled, num_axis_points)
    X, Y = np.meshgrid(x, y)
    
    # Create states
    states = equilibrium_states.clone().unsqueeze(0).repeat(num_axis_points**2, 1)
    states[:, idx1] = torch.from_numpy(X.flatten()).float()
    # scale velocity by 0.05 as happens during training
    states[:, idx2] = torch.from_numpy(Y.flatten()).float() * 0.05
    with torch.no_grad():
        V_values = trainer.get_lyapunov_output(states.to(device), equilibrium_states.to(device)).cpu().numpy()
    V = V_values.reshape(num_axis_points, num_axis_points)
    
    Plot3D(X, Y, V, filename=filename, xlabel=label1, ylabel=label2)
    print("Plot saved successfully in `{}`.".format(filename))


def plot_g1_balance_lyapunov_function(trainer, X_0, load_path, png_path='lyapunov.png', idx1=0, idx2=1):
    trainer.load_model(load_path)
    n_inputs = X_0.shape[0]

    # offsets for joint naming
    joint_pos_start = 45 + 4*23 if n_inputs == 390 else 0
    joint_pos_end = joint_pos_start + 23
    joint_vel_start = joint_pos_end + 4*23 if n_inputs == 390 else joint_pos_end
    offset_1 = joint_pos_start
    offset_2 = joint_vel_start - 23

    pos_joint_names = [name + '_pos' for name in JOINT_NAMES]
    vel_joint_names = [name + '_vel' for name in JOINT_NAMES]
    joint_names = pos_joint_names + vel_joint_names

    # Plot 3D Lyapunov function
    # x is position
    x_min, x_max = JOINT_LIMITS[idx1 - offset_1]
    # y is velocity
    y_min, y_max = JOINT_LIMITS[idx2 - offset_2]
    label1 = joint_names[idx1 - offset_1]
    label2 = joint_names[idx2 - offset_2]
    plot_lyapunov_3d(trainer, X_0, png_path, 
                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                     idx1=idx1, idx2=idx2, device=trainer.device,
                     label1=label1, label2=label2, factor_1=0.5, factor_2=0.05)
    