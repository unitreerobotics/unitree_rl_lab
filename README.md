
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=it">Itapano</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=unitreerobotics&project=unitree_rl_lab&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
  </details>
</div>

# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

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
    python -m pip install -e source/unitree_rl_lab
    ```
- Download unitree usd files

  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```
- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    python scripts/list_envs.py
    ```
  - Running a task:

    ```bash
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity-Play
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd deploy/robots/g1 # or other robots
mkdir build && cd build
cmake .. && make
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
```

```bash
cd unitree_rl_lab/deploy/robots/g1/build
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
