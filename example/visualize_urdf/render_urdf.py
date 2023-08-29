# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from pathlib import Path

import numpy as np

from sim_web_visualizer import MeshCatVisualizerBase
from sim_web_visualizer.parser.yourdfpy import URDF


def main():
    viz = MeshCatVisualizerBase(port=6000, host="localhost")

    # asset_path = Path(__file__).parent / "kuka_allegro_description"
    # urdf_path = asset_path / "kuka_allegro.urdf"
    asset_path = Path(__file__).parent / "piano"
    urdf_path = asset_path / "piano.urdf"

    # Load a yourdfpy instance only for forward kinematics computation
    robot = URDF.load(str(urdf_path), build_tree=True)
    dof = robot.num_dofs

    # Load robot URDF into the visualizer
    viz.viz["/URDF"].delete()
    asset_resource = viz.dry_load_asset(str(urdf_path), collapse_fixed_joints=False)
    viz.load_asset_resources(asset_resource, f"/URDF", scale=1.0)
    urdf_viz = viz.viz["/URDF"]
    qpos = np.zeros(dof)

    try:
        while True:
            robot.update_kinematics(qpos)
            delta_qpos = np.random.randn(dof) * 0.000
            qpos += delta_qpos
            for link_name, link in robot.link_map.items():
                pose = robot.get_link_global_transform(link_name)
                urdf_viz[link_name].set_transform(pose)
    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down.\n")


if __name__ == "__main__":
    main()
