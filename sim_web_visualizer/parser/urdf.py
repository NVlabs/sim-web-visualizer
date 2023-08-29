# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

# Utils for load URDF for meshcat server

import meshcat.geometry as g
import numpy as np
import transforms3d
import trimesh

import sim_web_visualizer.parser.yourdfpy as urdf
from sim_web_visualizer.parser.mesh_parser import load_mesh, get_trimesh_geometry_material, rgb_to_hex, AssetResource


def load_urdf_with_yourdfpy(
    urdf_path: str, collapse_fixed_joints: bool, replace_cylinder_with_capsule=False, use_mesh_materials=False
) -> AssetResource:
    robot = urdf.URDF.load(urdf_path)
    material_map = robot._material_map if not use_mesh_materials else {}

    # Dry run data for faster loading
    offline_data_dict = {}
    offline_pose_dict = {}

    # Deal with collapsed link
    link_root_map = {robot.base_link: robot.base_link}
    link_pose_map = {robot.base_link: np.eye(4)}
    if collapse_fixed_joints:
        # Sort joint map
        _link_cache = [robot.base_link]
        _joint_unsorted = list(robot.joint_map.keys())
        _joint_sorted = []
        while len(_joint_sorted) < len(_joint_unsorted):
            for joint_name in _joint_unsorted:
                if joint_name in _joint_sorted:
                    continue
                if robot.joint_map[joint_name].parent in _link_cache:
                    _link_cache.append(robot.joint_map[joint_name].child)
                    _joint_sorted.append(joint_name)

        # Build joint tree
        for joint_name in _joint_sorted:
            joint_info = robot.joint_map[joint_name]
            parent = joint_info.parent
            child = joint_info.child
            if joint_info.type == "fixed":
                assert parent in link_root_map
                if parent in link_root_map:
                    link_root_map[child] = link_root_map[parent]
                    link_pose_map[child] = link_pose_map[parent] @ joint_info.origin
            else:
                link_root_map[child] = child
                link_pose_map[child] = np.eye(4)

    link_mesh_count = {}
    for link_name, link_info in robot.link_map.items():
        for visual in link_info.visuals:
            geom = visual.geometry
            visual_pose = visual.origin if visual.origin is not None else np.eye(4)
            mats = []
            geometries = []
            poses = []
            urdf_rgba = None

            # Parse rgba from URDF material
            if visual.material is not None:
                urdf_rgba = np.clip(visual.material.color.rgba, 0, 1) if visual.material.color is not None else None
                # Check whether the material is pre-defined in the URDF if no color information
                if urdf_rgba is None:
                    if visual.material.name in material_map:
                        urdf_color = material_map[visual.material.name].color
                        urdf_rgba = np.clip(urdf_color.rgba, 0, 1)

            if collapse_fixed_joints:
                visual_pose = link_pose_map[link_name] @ visual_pose

            # Mesh geometry
            if geom.mesh is not None:
                mesh_scale = geom.mesh.scale if geom.mesh.scale is not None else np.array([1, 1, 1])
                mesh_filename = robot.filename_handler(fname=geom.mesh.filename)
                if mesh_filename.lower().endswith("obj") or mesh_filename.lower().endswith("dae"):
                    trimesh_geom = trimesh.load(mesh_filename, ignore_broken=True)

                    assert mesh_filename.lower().endswith("obj") or isinstance(trimesh_geom, trimesh.Scene)

                    # Handle complex obj where multiple mesh are presented
                    if isinstance(trimesh_geom, trimesh.Scene):
                        for name, scene_geometry in trimesh_geom.geometry.items():
                            mesh = scene_geometry.visual.mesh
                            mesh.apply_scale(mesh_scale)
                            geometries.append(g.TriangularMeshGeometry(mesh.vertices, mesh.faces))
                            mats.append(get_trimesh_geometry_material(scene_geometry, urdf_rgba))
                            local_pose = visual_pose @ trimesh_geom.graph.get(name)[0]
                            poses.append(local_pose)
                    else:
                        exp_mesh = load_mesh(mesh_filename, mesh_scale, trimesh_geom)
                        geometries = [g.ObjMeshGeometry.from_stream(trimesh.util.wrap_as_stream(exp_mesh))]
                        mats = [get_trimesh_geometry_material(trimesh_geom, urdf_rgba)]
                        poses.append(visual_pose)
                elif mesh_filename.lower().endswith("stl"):
                    mesh = trimesh.load(mesh_filename, ignore_broken=True)
                    mesh.apply_scale(mesh_scale)
                    geometries = [g.TriangularMeshGeometry(mesh.vertices, mesh.faces)]
                    mats = [get_trimesh_geometry_material(mesh, urdf_rgba)]
                    poses.append(visual_pose)
                elif mesh_filename.lower().endswith("glb"):
                    scene = trimesh.load(mesh_filename, ignore_broken=True, force="scene")
                    geom_name_map = {scene.graph[node_name][1]: node_name for node_name in scene.graph.nodes}
                    for name, scene_geometry in scene.geometry.items():
                        mesh = scene_geometry.visual.mesh
                        mesh.apply_scale(mesh_scale)
                        geometries.append(g.TriangularMeshGeometry(mesh.vertices, mesh.faces))
                        mats.append(get_trimesh_geometry_material(scene_geometry, urdf_rgba))
                        poses.append(visual_pose @ scene.graph.get(geom_name_map[name])[0])
                else:
                    raise NotImplementedError
            # Primitive geometry
            else:
                if geom.sphere is not None:
                    geometries = [g.Sphere(radius=geom.sphere.radius)]
                elif geom.box is not None:
                    geometries = [g.Box(geom.box.size)]
                elif geom.cylinder is not None and not replace_cylinder_with_capsule:
                    geometries = [g.Cylinder(geom.cylinder.length, geom.cylinder.radius)]
                    visual_pose[:3, :3] = visual_pose[:3, :3] @ transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
                elif geom.cylinder is not None and replace_cylinder_with_capsule:
                    geometries = [g.Capsule(geom.cylinder.radius, geom.cylinder.length)]
                    visual_pose[:3, :3] = visual_pose[:3, :3] @ transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
                else:
                    raise NotImplementedError

                poses.append(visual_pose)
                if urdf_rgba is not None:
                    rgb = (urdf_rgba[:3] * 255).astype(np.uint8)
                    mats.append(g.MeshPhongMaterial(color=rgb_to_hex(rgb), opacity=urdf_rgba[3]))
                else:
                    mats.append(None)

            # If collapsed, get the root link name for each fixed joint chain
            if collapse_fixed_joints:
                meshcat_link_name = link_root_map[link_name]
            else:
                meshcat_link_name = link_name
            # Count geom name for each link
            if meshcat_link_name not in link_mesh_count:
                link_mesh_count[meshcat_link_name] = 0
            else:
                link_mesh_count[meshcat_link_name] += 1
            geom_id = link_mesh_count[meshcat_link_name]

            # Add geometries into the visualizer or pack them offline for future use
            for each_geometry, each_mat, each_pose in zip(geometries, mats, poses):
                offline_data_dict[f"{meshcat_link_name}/{geom_id}"] = (each_geometry, each_mat)
                offline_pose_dict[f"{meshcat_link_name}/{geom_id}"] = each_pose
                geom_id += 1
            link_mesh_count[meshcat_link_name] = geom_id - 1

    resource = AssetResource(
        filename=urdf_path,
        visual_data=offline_data_dict,
        pose_data=offline_pose_dict,
    )
    return resource
