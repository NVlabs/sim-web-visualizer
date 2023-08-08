# Web-Based Visualizer for Simulation Environments

|  IsaacGym Simulator  | ![](doc/isaacgymenv/allegro.png) | ![](doc/isaacgymenv/cabinet.png) | ![](doc/isaacgymenv/dog.png) | ![](doc/isaacgymenv/kuka.png) | ![](doc/isaacgymenv/trifinger.png) |
|:--------------------:|:--------------------------------:|:--------------------------------:|:----------------------------:|:-----------------------------:|-----------------------------------:|
| **SAPIEN Simulator** | ![](doc/maniskill/assembly.png)  |   ![](doc/maniskill/avoid.png)   | ![](doc/maniskill/chair.png) | ![](doc/maniskill/insert.png) |         ![](doc/maniskill/ycb.png) |
| **Static URDF File** |     ![](doc/urdf/chair.png)      |     ![](doc/urdf/drawer.png)     |     ![](doc/urdf/dj.png)     |    ![](doc/urdf/piano.png)    |     ![](doc/urdf/kuka_allegro.png) |

This repository hosts a browser-based 3D viewer for physical simulators. It offers users the ability to observe
simulations directly within their web browser, as an alternative to the default visualizer that comes with the
simulator.

The main feature of this repo is that you only need to modify a server lines of your code to port the
visualization on the default simulator viewer to the web visualizer. This feature is especially useful for visualizing
simulation on a headless server. For example, train and visualize the IsaacGym tasks
inside [jupyter notebook](example/isaacgym/train_isaacgym_remote_server.ipynb) on a remote server.

## Installation

First install ZeroMQ libraries on your system:

```shell
apt install libzmq3-dev # For ubuntu
brew install zmq # For Mac
```

```shell
# git clone this repo
cd sim-web-visualier && git submodule update --init --recursive
pip install -e .
```

## Examples

### IsaacGym Web Visualizer

Check the [IsaacGym Example](example/isaacgym/README.md) for information on running the
Web Visualizer on [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

![isaac](doc/isaac.gif)

### SAPIEN Web Visualizer

Check the [SAPIEN Example](example/sapien/README.md) for information on running the
Web Visualizer on [ManiSkill2](https://github.com/haosulab/ManiSkill2).

### URDF Web Visualizer

Check the [URDF Example](example/visualize_urdf/README.md) for information to visualize a static URDF file on Web
Visualizer.

## Citing This Repo

This repository is a part of the [AnyTeleop Project](anyteleop.com/). If you use this work, kindly reference it
as:

```shell
@inproceedings{qin2023anyteleop,
  title     = {AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System},
  author    = {Qin, Yuzhe and Yang, Wei and Huang, Binghao and Van Wyk, Karl and Su, Hao and Wang, Xiaolong and Chao, Yu-Wei and Fox, Dieter},
  booktitle = {Robotics: Science and Systems},
  year      = {2023}
}
```

## Acknowledgments

This repository is developed upon the outstanding [MeshCat](https://github.com/rdeits/meshcat) project. We extend our
appreciation to the developers and custodians of MeshCat.

## License and Disclaimer

Web-Based Visualizer for Simulation Environments is released under the [MIT License](LICENSE).
