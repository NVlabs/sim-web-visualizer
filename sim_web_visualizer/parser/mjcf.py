# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from pathlib import Path
from typing import Dict, List

import meshcat.geometry as g
import meshcat.transformations as transformation
import numpy as np
import transforms3d
import trimesh
from lxml import etree
from meshcat.visualizer import Visualizer

from sim_web_visualizer.parser.mesh_parser import rgb_to_hex, AssetResource
from sim_web_visualizer.utils.rotation_utils import compute_vector_rotation


def load_mjcf_with_dmc(filename: str, collapse_fixed_joints: bool) -> AssetResource:
    from dm_control import mjcf

    # The mjcf file by IsaacGym does not follow the convention of mujoco mjcf precisely
    # We need to handle it separately when the mjcf file is not valid by normal mjcf parser
    isaac_mjcf = False
    file_root = Path(filename).parent

    def get_geom_pose(geom: mjcf.Element) -> np.ndarray:
        assert geom.tag == "geom"
        pos = geom.pos if geom.pos is not None else np.array([0, 0, 0])
        if geom.quat is not None:
            pose = transformation.quaternion_matrix(geom.quat)
        elif geom.euler is not None:
            # TODO: eulerseq
            pose = transformation.euler_matrix(*geom.euler)
        else:
            pose = np.eye(4)
        pose[:3, 3] = pos
        return pose

    try:
        model = mjcf.from_path(filename)
    except KeyError:
        isaac_mjcf = True
        tree = etree.parse(filename)
        root = tree.getroot()
        invalid_includes = root.findall("*/include")
        for include in invalid_includes:
            parent = include.getparent()
            file: str = include.get("file")
            file_path = file_root / file
            child_xml = etree.parse(str(file_path)).getroot().getchildren()
            parent.remove(include)
            parent.extend(child_xml)

        xml_string = etree.tostring(tree)
        model = mjcf.from_xml_string(xml_string, model_dir=str(file_root))

    # Dry run data for faster loading
    offline_data_dict = {}
    offline_pose_dict = {}

    # Substitute geom with default values
    for geom in model.find_all("geom"):
        mjcf.commit_defaults(geom)

    # Build data structure for link to keep all its geometries and assign a link_name if not specified
    body_geom_map: Dict[str, List[mjcf.Element]] = {}
    body_name_count = 0
    for body in model.find_all("body"):
        if body.name is not None:
            body_name = body.name
        else:
            body_name = f"body_{body_name_count}"
            body.name = body_name
            body_name_count += 1

        temp_body_geom_list = []
        for geom in body.find_all("geom", immediate_children_only=True):
            temp_body_geom_list.append(geom)
        body_geom_map[body_name] = temp_body_geom_list

    # Find the real parent of a body if collapse_fixed_joints is True and the body is attached to a fixed joint
    root_link_list = []
    if collapse_fixed_joints:
        for body in model.find_all("body"):
            if body.parent.tag == "worldbody":
                root_link_list.append(body)
                continue

            # If no joints is attached to the body, then it is a fixed joint
            joints = body.joint
            if len(joints) == 0:
                parent = body.parent
                while parent not in root_link_list and len(parent.joints) == 0:
                    parent = parent.parent

                geom_list = body_geom_map.pop(body.name)
                body_geom_map[parent.name].extend(geom_list)
            else:
                root_link_list.append(body)

    for link_name, geoms in body_geom_map.items():
        if len(geoms) == 0:
            continue

        for geom_id, geom in enumerate(geoms):  # type: int, mjcf.Element
            geom_type = geom.type
            size = geom.size if geom.size is not None else np.array([1, 1, 1])
            from_to = geom.fromto if geom.size is not None else None
            rgba = geom.rgba if geom.rgba is not None else np.array([0.5, 0.5, 0.5, 1])

            pose = get_geom_pose(geom)
            mat = None

            if isaac_mjcf:
                if geom.contype ^ geom.conaffinity:
                    continue

            if geom_type == "capsule":
                # Note that in MuJoCo, rotation axis is z-axis while it is y-axis in three js
                radius = size[0]
                if from_to is not None:
                    vector = from_to[3:6] - from_to[0:3]
                    length = np.linalg.norm(vector)
                    center = (from_to[3:6] + from_to[0:3]) / 2
                else:
                    vector = np.array([0, 0, 1])
                    length = size[1] * 2  # MuJoCo mjcf uses half-length but three js uses full-length
                    center = np.array([0, 0, 0])

                geometry = g.Capsule(radius=radius, length=length)
                capsule_transformation = compute_vector_rotation(np.array([0, 1.0, 0]), vector.astype(float))
                capsule_transformation[:3, 3] = center
                pose = pose @ capsule_transformation

            elif geom_type == "sphere":
                geometry = g.Sphere(radius=float(size[0]))
            elif geom_type == "box":
                geometry = g.Box(lengths=size * 2)
            elif geom_type == "cylinder":
                if from_to is None:
                    geometry = g.Cylinder(height=size[1] * 2, radius=size[0])
                    pose[:3, :3] = pose[:3, :3] @ transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
                else:
                    vector = from_to[3:6] - from_to[0:3]
                    length = np.linalg.norm(vector)
                    center = (from_to[3:6] + from_to[0:3]) / 2
                    geometry = g.Cylinder(height=length, radius=size[0])
                    cylinder_transformation = compute_vector_rotation(np.array([0, 1.0, 0]), vector.astype(float))
                    pose = pose @ cylinder_transformation
                    cylinder_transformation[:3, 3] = center

            elif geom_type == "mesh":
                mesh = geom.mesh
                scale = mesh.scale if mesh.scale is not None else np.array([1.0, 1.0, 1.0])
                if geom.material is not None:
                    material = geom.material
                    rgba = (np.clip(material.rgba, 0, 1) * 255).astype(int)
                    mat = g.MeshLambertMaterial(
                        reflectivity=material.reflectance,
                        color=rgb_to_hex(rgba[:3]),
                        opacity=rgba[3] / 255.0,
                    )

                if mesh.file.extension == ".stl":
                    stl_dict = trimesh.exchange.stl.load_stl_binary(trimesh.util.wrap_as_stream(mesh.file.contents))
                    v = stl_dict["vertices"] * scale[None, :]
                    f = stl_dict["faces"]
                    geometry = g.TriangularMeshGeometry(v, f)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            if mat is None:
                rgba = (np.clip(rgba, 0, 1) * 255).astype(int)
                mat = g.MeshLambertMaterial(color=rgb_to_hex(rgba[:3]), opacity=rgba[3] / 255.0)

            if isinstance(geometry, List):
                sub_geom_id = 0
                for each_geometry, each_mat, each_pose in zip(geometry, mat, pose):
                    offline_data_dict[f"{link_name}/{geom_id}/{sub_geom_id}"] = (each_geometry, each_mat)
                    offline_pose_dict[f"{link_name}/{geom_id}/{sub_geom_id}"] = each_pose
                    geom_id += 1
            else:
                offline_data_dict[f"{link_name}/{geom_id}"] = (geometry, mat)
                offline_pose_dict[f"{link_name}/{geom_id}"] = pose

    return AssetResource(filename, offline_data_dict, offline_pose_dict)


def load_mjcf_into_viewer_kinpy(filename: str, viewer: Visualizer, collapse_fixed_joints: bool, dry_run=False):
    import kinpy as kp

    chain = kp.build_chain_from_mjcf(open(filename).read())

    # Dry run data for faster loading
    offline_data_dict = {}
    offline_pose_dict = {}

    original_visual_maps = chain.visuals_map()
    visual_maps = {k: v for k, v in original_visual_maps.items() if not k.endswith("_child")}
    child_maps = {k[:-6]: v for k, v in original_visual_maps.items() if k.endswith("child")}
    visual_maps.update(child_maps)

    for link, geoms in visual_maps.items():
        if len(geoms) == 0:
            continue

        for geom_id, geom in enumerate(geoms):
            geom_type = geom.geom_type
            params = geom.geom_param
            pose = transformation.quaternion_matrix(geom.offset.rot)
            pose[:3, 3] = geom.offset.pos
            if geom_type == "capsule":
                # Note that in MuJoCo, rotation axis is z-axis while it is y-axis in three js
                if isinstance(params, tuple):
                    vector = params[1][3:6] - params[1][0:3]
                    length = np.linalg.norm(vector)
                    radius = params[0]
                    center = (params[1][:3] + params[1][3:6]) / 2
                else:
                    raise NotImplementedError

                geometry = g.Capsule(radius=radius, length=length)
                capsule_transformation = compute_vector_rotation(np.array([0, 1.0, 0]), vector)
                capsule_transformation[:3, 3] = center
                pose = pose @ capsule_transformation

            elif geom_type == "sphere":
                geometry = g.Sphere(radius=params)
            else:
                raise NotImplementedError

            if dry_run:
                offline_data_dict[f"{link}/{geom_id}"] = (geometry, g.MeshPhongMaterial())
                offline_pose_dict[f"{link}/{geom_id}"] = pose
            else:
                viewer[f"{link}/{geom_id}"].set_object(geometry)
                viewer[f"{link}/{geom_id}"].set_transform(pose)

    if dry_run:
        return offline_data_dict, offline_pose_dict
