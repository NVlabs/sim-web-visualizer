# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


"""
Port ManiSkill2 demonstration replay visualization to web visualizer
# Reference: https://github.com/haosulab/ManiSkill2/tree/main/examples/tutorials/imitation-learning

"""


import argparse
import os
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
from mani_skill2.envs import sapien_env
from mani_skill2.utils.io_utils import load_json
from tqdm.auto import tqdm

################################################################################################
# The following code is only used for visualizer and not presented in the original ManiSkill
################################################################################################
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from sim_web_visualizer import bind_visualizer_to_sapien_scene, create_sapien_visualizer


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


start_zmq_server_as_subprocess()
# Set to True if you want to keep both the original viewer and the web visualizer. A display is needed for True
keep_on_screen_renderer = True

create_sapien_visualizer(port=6000, host="localhost", keep_default_viewer=keep_on_screen_renderer)
sapien_env.BaseEnv._setup_scene = wrapped_setup_scene
sapien_env.BaseEnv._setup_viewer = wrapped_setup_viewer
################################################################################################
# End of visualizer code
################################################################################################


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of demonstrations to replay before exiting. By default will replay all demonstrations",
    )
    return parser.parse_args(args)


def main(args):
    pbar = tqdm(position=0, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    # Create a main env for replay
    env_kwargs = ori_env_kwargs.copy()
    env_kwargs["obs_mode"] = "state"
    env_kwargs[
        "render_mode"
    ] = "rgb_array"  # note this only affects the videos saved as RecordEpisode wrapper calls env.render
    env = gym.make(env_id, **env_kwargs).unwrapped
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

    # Prepare for recording
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    suffix = "{}.{}".format(env.obs_mode, env.control_mode)
    new_traj_name = ori_traj_name + "." + suffix

    episodes = json_data["episodes"][: args.count]
    n_ep = len(episodes)
    inds = np.arange(n_ep)
    inds = np.array_split(inds, 1)[0]

    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]
        seed = reset_kwargs.pop("seed")

        env.reset(seed=seed, options=reset_kwargs)
        env.render_human()

        # Original actions to replay
        ori_actions = ori_h5_file[traj_id]["actions"][:]

        # Original env states to replay
        ori_env_states = ori_h5_file[traj_id]["env_states"][1:]

        info = {}

        # Without conversion between control modes
        n = len(ori_actions)
        if pbar is not None:
            pbar.reset(total=n)
        for t, a in enumerate(ori_actions):
            if pbar is not None:
                pbar.update()
            env.render_human()
            env.set_state(ori_env_states[t])

        success = info.get("success", False)

    # Cleanup
    env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main(parse_args())
