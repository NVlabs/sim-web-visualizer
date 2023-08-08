# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


"""
Original mani-skill2 code without the web visualization
# Reference: https://haosulab.github.io/ManiSkill2/getting_started/quickstart.html

import gym
import mani_skill2.envs  # import to register all environments in gym

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_ee_delta_pose")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()  # a display is required to render
env.close()
"""

from typing import Optional

import gym
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


create_sapien_visualizer(port=6000, host="localhost", keep_default_viewer=True)
sapien_env.BaseEnv._setup_scene = wrapped_setup_scene

sapien_env.BaseEnv._setup_viewer = wrapped_setup_viewer

task_names = ["MoveBucket-v1", "PushChair-v1", "OpenCabinetDrawer-v1", "TurnFaucet-v0", "PandaAvoidObstacles-v0",
              "AssemblingKits-v0", "PlugCharger-v0", "PegInsertionSide-v0", "PickClutterYCB-v0", "PickSingleEGAD-v0",
              "StackCube-v0"]
control_mode = ["base_pd_joint_vel_arm_pd_joint_vel"] * 3 + ["pd_joint_delta_pos"] * 8
task_num = 1

env = gym.make(task_names[task_num], obs_mode="rgbd", control_mode=control_mode[task_num])
# print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(1)  # specify a seed for randomness

while True:
    obs = env.reset()
    done = False
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # env.render()  # a display is required to render

env.close()
