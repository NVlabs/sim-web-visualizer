# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from transforms3d.euler import euler2quat

from sim_web_visualizer.parser.yourdfpy import URDF
from sim_web_visualizer import create_sapien_visualizer, bind_visualizer_to_sapien_scene


def create_car(
    scene: sapien.Scene,
    body_size=(1.0, 0.5, 0.25),
    tire_radius=0.15,
    joint_friction=0.0,
    joint_damping=0.0,
    density=1.0,
) -> sapien.Articulation:
    # Code source: https://sapien.ucsd.edu/docs/2.2/tutorial/basic/create_articulations.html
    body_half_size = np.array(body_size) / 2
    shaft_half_size = np.array([tire_radius * 0.1, tire_radius * 0.1, body_size[2] * 0.1])
    rack_half_size = np.array([tire_radius * 0.1, body_half_size[1] * 2.0, tire_radius * 0.1])
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    # car body (root of the articulation)
    body: sapien.LinkBuilder = builder.create_link_builder()  # LinkBuilder is similar to ActorBuilder
    body.set_name("body")
    body.add_box_collision(half_size=body_half_size, density=density)
    body.add_box_visual(half_size=body_half_size, color=[0.8, 0.6, 0.4])

    # front steering shaft
    front_shaft = builder.create_link_builder(body)
    front_shaft.set_name("front_shaft")
    front_shaft.set_joint_name("front_shaft_joint")
    front_shaft.add_box_collision(half_size=shaft_half_size, density=density)
    front_shaft.add_box_visual(half_size=shaft_half_size, color=[0.6, 0.4, 0.8])
    # The x-axis of the joint frame is the rotation axis of a revolute joint.
    front_shaft.set_joint_properties(
        "revolute",
        limits=[[-np.deg2rad(15), np.deg2rad(15)]],  # joint limits (for each DoF)
        # pose_in_parent refers to the relative transformation from the parent frame to the joint frame
        pose_in_parent=sapien.Pose(
            p=[(body_half_size[0] - tire_radius), 0, -body_half_size[2]], q=euler2quat(0, -np.deg2rad(90), 0)
        ),
        # pose_in_child refers to the relative transformation from the child frame to the joint frame
        pose_in_child=sapien.Pose(p=[0.0, 0.0, shaft_half_size[2]], q=euler2quat(0, -np.deg2rad(90), 0)),
        friction=joint_friction,
        damping=joint_damping,
    )

    # back steering shaft (not drivable)
    back_shaft = builder.create_link_builder(body)
    back_shaft.set_name("back_shaft")
    back_shaft.set_joint_name("back_shaft_joint")
    back_shaft.add_box_collision(half_size=shaft_half_size, density=density)
    back_shaft.add_box_visual(half_size=shaft_half_size, color=[0.6, 0.4, 0.8])
    back_shaft.set_joint_properties(
        "fixed",
        limits=[],
        pose_in_parent=sapien.Pose(
            p=[-(body_half_size[0] - tire_radius), 0, -body_half_size[2]], q=euler2quat(0, -np.deg2rad(90), 0)
        ),
        pose_in_child=sapien.Pose(p=[0.0, 0.0, shaft_half_size[2]], q=euler2quat(0, -np.deg2rad(90), 0)),
        friction=joint_friction,
        damping=joint_damping,
    )

    # front wheels
    front_wheels = builder.create_link_builder(front_shaft)
    front_wheels.set_name("front_wheels")
    front_wheels.set_joint_name("front_gear")
    # rack
    front_wheels.add_box_collision(half_size=rack_half_size, density=density)
    front_wheels.add_box_visual(half_size=rack_half_size, color=[0.8, 0.4, 0.6])
    # left wheel
    front_wheels.add_sphere_collision(
        pose=sapien.Pose(p=[0.0, rack_half_size[1] + tire_radius, 0.0]), radius=tire_radius, density=density
    )
    front_wheels.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, rack_half_size[1] + tire_radius, 0.0]), radius=tire_radius, color=[0.4, 0.6, 0.8]
    )
    # right wheel
    front_wheels.add_sphere_collision(
        pose=sapien.Pose(p=[0.0, -(rack_half_size[1] + tire_radius), 0.0]), radius=tire_radius, density=density
    )
    front_wheels.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, -(rack_half_size[1] + tire_radius), 0.0]), radius=tire_radius, color=[0.4, 0.6, 0.8]
    )
    # gear
    front_wheels.set_joint_properties(
        "revolute",
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(
            p=[0.0, 0, -(shaft_half_size[2] + rack_half_size[2])], q=euler2quat(0, 0, np.deg2rad(90))
        ),
        pose_in_child=sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping,
    )

    # back wheels
    back_wheels = builder.create_link_builder(back_shaft)
    back_wheels.set_name("back_wheels")
    back_wheels.set_joint_name("back_gear")
    # rack
    back_wheels.add_box_collision(half_size=rack_half_size, density=density)
    back_wheels.add_box_visual(half_size=rack_half_size, color=[0.8, 0.4, 0.6])
    # left wheel
    back_wheels.add_sphere_collision(
        pose=sapien.Pose(p=[0.0, rack_half_size[1] + tire_radius, 0.0]), radius=tire_radius, density=density
    )
    back_wheels.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, rack_half_size[1] + tire_radius, 0.0]), radius=tire_radius, color=[0.4, 0.6, 0.8]
    )
    # right wheel
    back_wheels.add_sphere_collision(
        pose=sapien.Pose(p=[0.0, -(rack_half_size[1] + tire_radius), 0.0]), radius=tire_radius, density=density
    )
    back_wheels.add_sphere_visual(
        pose=sapien.Pose(p=[0.0, -(rack_half_size[1] + tire_radius), 0.0]), radius=tire_radius, color=[0.4, 0.6, 0.8]
    )
    # gear
    back_wheels.set_joint_properties(
        "revolute",
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(
            p=[0.0, 0, -(shaft_half_size[2] + rack_half_size[2])], q=euler2quat(0, 0, np.deg2rad(90))
        ),
        pose_in_child=sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping,
    )

    car = builder.build()
    car.set_name("car")
    return car


def main(filename):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    config = sapien.SceneConfig()
    config.gravity = np.array([0, 0, -9.8])
    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 125)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([1, -1, -1], [0.5, 0.5, 0.5])
    scene.add_point_light([2, 2, 2], [1, 1, 1])
    scene.add_point_light([2, -2, 2], [1, 1, 1])
    scene.add_point_light([-2, 0, 2], [1, 1, 1])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    create_sapien_visualizer(port=6000, host="localhost")
    scene = bind_visualizer_to_sapien_scene(scene, engine, renderer)

    viewer.set_camera_xyz(-1, 0, 1)
    viewer.set_camera_rpy(0, -0.8, 0)
    scene.add_ground(altitude=0)  # Add a ground

    # Load from URDF loader
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 0.8
    cabinet = loader.load("./1000/mobility_cvx.urdf")
    cabinet.set_pose(sapien.Pose([1, 0, 0.5]))

    # Load from actor build
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1.0, 0.0, 0.0])
    box = actor_builder.build(name="box")  # Add a box
    box.set_pose(sapien.Pose(p=[-1, 0, 0.5]))

    # Load from articulation builder
    car = create_car(scene)
    car.set_pose(sapien.Pose(p=[0.0, 2.0, 0.34]))

    print("Press q to quit......")

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename of the urdf you would like load.")
    args = parser.parse_args()
    main(args.filename)
