# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


"""
Original mani-skill2 code without the web visualization
# Reference: https://haosulab.github.io/ManiSkill2/getting_started/quickstart.html

import gymnasium as gym
import mani_skill2.envs

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="human")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for randomness
terminated, truncated = False, False
while not terminated and not truncated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # a display is required to render
env.close()
"""

from typing import Optional

import gymnasium as gym

# import to register all environments in gym
# noinspection PyUnresolvedReferences
import mani_skill2.envs  # pylint: disable=unused-import
import sapien.core as sapien
from mani_skill2.envs import sapien_env

from sim_web_visualizer import create_sapien_visualizer, bind_visualizer_to_sapien_scene


def wrapped_setup_scene(self: sapien_env.BaseEnv, scene_config: Optional[sapien.SceneConfig] = None):
    if scene_config is None:
        scene_config = self._get_default_scene_config()
    self._scene = self._engine.create_scene(scene_config)
    self._scene.set_timestep(1.0 / self._sim_freq)
    self._scene = bind_visualizer_to_sapien_scene(self._scene, self._engine, self._renderer)


def wrapped_setup_viewer(self):
    self._viewer.set_scene(self._scene._scene)
    self._viewer.scene = self._scene
    self._viewer.toggle_axes(False)
    self._viewer.toggle_camera_lines(False)


# Set to True if you want to keep both the original viewer and the web visualizer. A display is needed for True
keep_on_screen_renderer = False

create_sapien_visualizer(port=6000, host="localhost", keep_default_viewer=keep_on_screen_renderer)
sapien_env.BaseEnv._setup_scene = wrapped_setup_scene

sapien_env.BaseEnv._setup_viewer = wrapped_setup_viewer

task_names = [
    "MoveBucket-v1",
    "PushChair-v1",
    "OpenCabinetDrawer-v1",
    "TurnFaucet-v0",
    "PandaAvoidObstacles-v0",
    "AssemblingKits-v0",
    "PlugCharger-v0",
    "PegInsertionSide-v0",
    "PickClutterYCB-v0",
    "PickSingleEGAD-v0",
    "StackCube-v0",
]
control_mode = ["base_pd_joint_vel_arm_pd_joint_vel"] * 3 + ["pd_joint_delta_pos"] * 8

# You can try different task_num to visualize different tasks
task_num = 7

env = gym.make(task_names[task_num], obs_mode="rgbd", control_mode=control_mode[task_num])
# print("Observation space", env.observation_space)
print("Action space", env.action_space)

while True:
    try:
        obs = env.reset()
        done = False
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if keep_on_screen_renderer:
                env.render()  # a display is required to render
    except KeyboardInterrupt:
        break

env.close()
