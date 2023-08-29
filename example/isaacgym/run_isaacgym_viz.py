# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import os
import sys

# noinspection PyUnresolvedReferences
import isaacgym  # pylint: disable=unused-import
import torch
from isaacgym import gymapi

from sim_web_visualizer.isaac_visualizer_client import create_isaac_visualizer, bind_visualizer_to_gym, set_gpu_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "IsaacGymEnvs"))
import isaacgymenvs
from isaacgymenvs.tasks.base import vec_task


def wrapped_create_sim(
    self: vec_task.VecTask, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams
):
    sim = vec_task._create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    self.gym = bind_visualizer_to_gym(self.gym, sim)
    set_gpu_pipeline(sim_params.use_gpu_pipeline)
    return sim


# Reload VecTask function to create a hook for sim_web_visualizer
vec_task.VecTask.create_sim = wrapped_create_sim

# Create web visualizer
create_isaac_visualizer(port=6000, host="localhost", keep_default_viewer=True, max_env=4)

# Create the environment and step the simulation as normal
device = "cuda"  # or "cpu"
num_env = 8
rl_device = device
envs = isaacgymenvs.make(
    seed=0,
    task="AllegroHand",
    num_envs=num_env,
    sim_device=device,
    rl_device=device,
    graphics_device_id=0,
    headless=False,  # Need to be False if even you are using a headless server.
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

try:
    while True:
        envs.reset()
        for _ in range(500):
            obs, reward, done, info = envs.step(torch.rand((num_env,) + envs.action_space.shape, device=device))
except KeyboardInterrupt:
    print("Keyboard interrupt, shutting down.\n")
