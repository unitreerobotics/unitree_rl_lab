# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

To run the policy with [ExecuTorch](https://github.com/pytorch/executorch) instead of ONNX Runtime, build with `-DUSE_EXECUTORCH=ON`:

- The controller then loads `exported/policy.pte` instead of `exported/policy.onnx`, from each `policy_dir` in `config.yaml`. When `policy_dir` is a training log folder, the newest run with an `exported` folder is used, so put the `.pte` there.
- The program takes the same inputs as `policy.onnx`. It stores their names, in the order `forward()` takes them, in an `input_names` constant method, and uses the XNNPACK backend that the controllers link. The G1 policies in this repo come only as `policy.onnx`, so with the default `config.yaml` each enabled policy needs a `.pte` first.
- The `export_policy_to_executorch` exporter proposed for rsl_rl in [rsl_rl#271](https://github.com/leggedrobotics/rsl_rl/pull/271) writes such programs, but it needs rsl_rl 5, which is newer than the rsl_rl that Isaac Lab 2.3 installs. To export a policy trained here with ExecuTorch directly, use a separate Python environment, not the Isaac Lab one: `pip install executorch==1.6.0` brings torch 2.15.0, which would replace the torch that Isaac Lab uses. Rebuild the actor network there and load its weights from the training checkpoint (`num_obs` and `num_actions` are the sizes of the input and output of `policy.onnx`):

  ```python
  import torch
  from torch import nn
  from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
  from executorch.exir import to_edge_transform_and_lower

  # The actor from rsl_rl_ppo_cfg.py: actor_hidden_dims=[512, 256, 128], activation="elu", no observation normalization.
  policy = nn.Sequential(
      nn.Linear(num_obs, 512), nn.ELU(),
      nn.Linear(512, 256), nn.ELU(),
      nn.Linear(256, 128), nn.ELU(),
      nn.Linear(128, num_actions),
  )
  state = torch.load("model_<iteration>.pt", map_location="cpu")["model_state_dict"]
  policy.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
  policy.eval()

  program = to_edge_transform_and_lower(
      torch.export.export(policy, (torch.zeros(1, num_obs),)),
      partitioner=[XnnpackPartitioner()],
      constant_methods={"input_names": ["obs"]},
  ).to_executorch()
  open("exported/policy.pte", "wb").write(program.buffer)
  ```

- The build still needs ONNX Runtime, as the default build does.
- To build, the controller only needs the C++ library inside the ExecuTorch Python package, so install it without its dependencies, in its own Python environment (ExecuTorch 1.6.0 needs Python 3.11 or newer). The controller loads the library from that environment every time it starts, so keep the environment. The build needs CMake 3.28 or newer.

```bash
pip install --no-deps executorch==1.6.0
cmake .. -DUSE_EXECUTORCH=ON -DCMAKE_PREFIX_PATH="$(python -c 'import executorch, pathlib; print(pathlib.Path(executorch.__path__[0]) / "share" / "cmake")')"
make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
