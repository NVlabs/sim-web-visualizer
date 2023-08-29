# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from pathlib import Path
from typing import Optional

import meshcat
import numpy as np
import trimesh
from IPython.display import HTML
from meshcat import geometry as g

from sim_web_visualizer.parser.mesh_parser import get_trimesh_geometry_material, AssetResource
from sim_web_visualizer.parser.mjcf import load_mjcf_with_dmc
from sim_web_visualizer.parser.urdf import load_urdf_with_yourdfpy


def rgb_to_hex(rgb):
    rgb = (rgb * 255).astype(np.uint8)
    return "0x{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


class MeshCatVisualizerBase:
    def __init__(self, port: Optional[int] = None, host="localhost", continuous_scene=False):
        if port is not None:
            zmq_url = f"tcp://{host}:{port}"
        else:
            zmq_url = None
        self.viz = meshcat.Visualizer(zmq_url=zmq_url)
        if not continuous_scene:
            self.delete_all()
            self.add_default_scene_elements()

    def add_default_scene_elements(self):
        # Set initial color and light
        self.viz["/Background"].set_property("bottom_color", [0.8, 0.8, 0.8])
        self.viz["/Background"].set_property("top_color", [0.6, 0.6, 0.8])
        self.viz["/Lights/AmbientLight/<object>"].set_property("intensity", 0.8)
        self.viz["/Lights/PointLightNegativeX"].delete()
        self.viz["/Lights/PointLightPositiveX"].delete()

    def delete_all(self):
        print("Deleting all previous scene asset.")
        self.viz["/Teleop"].delete()
        self.viz["/Sim"].delete()

    @staticmethod
    def dry_load_asset(
        filename, collapse_fixed_joints: bool, replace_cylinder_with_capsule=False, use_mesh_materials=False
    ) -> AssetResource:
        if filename.endswith(".urdf"):
            resource = load_urdf_with_yourdfpy(
                filename,
                collapse_fixed_joints,
                replace_cylinder_with_capsule=replace_cylinder_with_capsule,
                use_mesh_materials=use_mesh_materials,
            )
        elif filename.endswith(".xml"):
            resource = load_mjcf_with_dmc(filename, collapse_fixed_joints)
        else:
            raise ValueError(f"Invalid file type: {filename}")

        return resource

    def load_asset_resources(self, resource: AssetResource, root_path: str, scale: float = 1.0):
        root_path = root_path.strip("/")
        for path, data in resource.visual_data.items():
            geom, mat = data[0], data[1]
            pose = resource.pose_data[path]
            pose[:3] *= scale
            self.viz[f"/{root_path}/{path}"].set_object(geom, mat)
            self.viz[f"/{root_path}/{path}"].set_transform(pose)

    # def create_operational_space_viz(self, hand_pose_vec: np.ndarray, cam2operator_pose_vec: np.ndarray, root_path: str,
    #                                  near=0.5, far=1, scale=1, fov=np.array([1.518, 1.012])):
    #     root_path = root_path.rstrip("/")
    #     viz = self.viz[root_path]
    #
    #     mat_cam2hand = tf.quaternion_matrix(cam2operator_pose_vec[3:7])
    #     mat_cam2hand[:3, 3] = cam2operator_pose_vec[0:3]
    #
    #     mat_hand2world = tf.quaternion_matrix(hand_pose_vec[3:7])
    #     mat_hand2world[:3, 3] = hand_pose_vec[0:3]
    #     mat_cam2world = mat_hand2world @ mat_cam2hand
    #     # operator2use = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    #     # mat_cam2world[:3, :3] = operator2use @ mat_cam2world[:3, :3]
    #     # mat_cam2world[:3, 3] -= np.array([1.5, 0, 0.5])
    #
    #     xy_scale = np.tan(fov / 2)
    #     cam_frustum_points = np.array([
    #         [xy_scale[0], xy_scale[1], 1],
    #         [-xy_scale[0], xy_scale[1], 1],
    #         [-xy_scale[0], -xy_scale[1], 1],
    #         [xy_scale[0], -xy_scale[1], 1],
    #     ])
    #     near_points = cam_frustum_points * near * scale
    #     far_points = cam_frustum_points * far * scale
    #
    #     class Points(g.Object):
    #         _type = u"Points"
    #
    #     pos_order = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7])
    #     point_pos = np.concatenate([near_points, far_points], axis=0)
    #     color_order = (pos_order >= 4).astype(int)
    #     point_color = np.array([[0.6, 1, 0.6], [0.2, 1, 0.2]])
    #     points = g.TriangularMeshGeometry(vertices=point_pos.astype(np.float32),
    #                                       # color=point_color[color_order].astype(np.float32),
    #                                       faces=np.array([[0]]).astype(np.float32))
    #     # material = g.LineBasicMaterial(vertexColors=True)
    #     # viz["lines"].set_object(g.LineSegments(points, material))
    #     # points_material = g.PointsMaterial()
    #     points_material = None
    #     viz["points"].set_object(Points(points), points_material)
    #     viz.set_transform(mat_cam2world)

    def create_coordinate_axis(self, pose_mat: np.ndarray, root_path: str, scale=1.0, opacity=1.0, sphere_radius=0.0):
        asset_path = Path(__file__).parent / "assets"
        axis_geom_path = (asset_path / "axis_geom" / "xyz_axis.obj").absolute()
        root_path = root_path.rstrip("/")
        scene = trimesh.load((str(axis_geom_path)), force="scene")
        viz = self.viz[root_path]

        for name, scene_geometry in scene.geometry.items():
            mesh = scene_geometry.visual.mesh
            mesh.apply_scale(scale)
            mat = get_trimesh_geometry_material(scene_geometry)
            mat.opacity = opacity
            mat.reflectivity = 0.0
            viz[name].set_object(g.TriangularMeshGeometry(mesh.vertices, mesh.faces), mat)
            viz[name].set_transform(scene.graph.get(name)[0])

        if sphere_radius > 1e-6:
            sphere = g.Sphere(radius=sphere_radius)
            mat = g.MeshPhongMaterial(color=rgb_to_hex(np.array([0.5, 0.5, 0.5])), opacity=opacity)
            viz["sphere"].set_object(sphere, mat)

        viz.set_transform(pose_mat)

    @staticmethod
    def jupyter_cell(url="tcp://127.0.0.1:6000", height=400):
        """
        Render the visualizer in a jupyter notebook or jupyterlab cell.

        For this to work, it should be the very last command in the given jupyter
        cell.
        """
        return HTML(
            """
            <div style="height: {height}px; width: 100%; overflow-x: auto; overflow-y: hidden; resize: both">
            <iframe src="{url}" style="width: 100%; height: 100%; border: none"></iframe>
            </div>
            """.format(
                url=url, height=height
            )
        )
