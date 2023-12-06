# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple

import meshcat
import meshcat.geometry as g
import numpy as np
import sapien.core as sapien
import trimesh

from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase, AssetResource
from sim_web_visualizer.parser.mesh_parser import get_trimesh_geometry_material, load_mesh
from sim_web_visualizer.parser.mesh_parser import rgb_to_hex


class MimicScene:
    def __init__(self, scene: sapien.Scene):
        self._scene = scene
        self.methods: Dict[str, Callable] = {}

    def add_method(self, name: str, fn: Callable):
        self.methods[name] = fn

    def __getattribute__(self, item):
        if item in ["add_method", "methods", "_scene"]:
            return super().__getattribute__(item)
        if item in self.methods:
            return self.methods[item]
        else:
            return self._scene.__getattribute__(item)


class MimicEntity:
    def __init__(self, sapien_entity):
        self.entity = sapien_entity
        self.methods: Dict[str, Callable] = {}

    def add_method(self, name: str, fn: Callable):
        self.methods[name] = fn

    def __getattribute__(self, item):
        if item in ["add_method", "methods", "entity"]:
            return super().__getattribute__(item)
        if item in self.methods:
            return self.methods[item]
        else:
            return self.entity.__getattribute__(item)

    def __setattr__(self, key, value):
        if key not in ["add_method", "methods", "entity"]:
            setattr(self.entity, key, value)
        else:
            super().__setattr__(key, value)


def visual2geom_mat(
    visual: sapien.VisualRecord, pose: np.ndarray
) -> Tuple[List[g.Geometry], List[g.Material], List[np.ndarray]]:
    scale = visual.scale.astype(float)
    material = visual.material
    if visual.type == "Box":
        geoms = [g.Box(scale * 2)]
    elif visual.type == "Sphere":
        geoms = [g.Sphere(visual.radius)]
    elif visual.type == "Capsule":
        geoms = [g.Capsule(radius=visual.radius, length=visual.length)]
    elif visual.type == "File":
        trimesh_geom = trimesh.load(visual.filename, ignore_broken=True)
        original_pose = pose
        poses = []
        mats = []
        geoms = []
        default_rgba = np.array(material.base_color) if material is not None else None
        if isinstance(trimesh_geom, trimesh.Scene):
            for name, scene_geometry in trimesh_geom.geometry.items():
                mesh = scene_geometry.visual.mesh
                mesh.apply_scale(scale)
                geoms.append(g.TriangularMeshGeometry(mesh.vertices, mesh.faces))
                mats.append(get_trimesh_geometry_material(scene_geometry, default_rgba=default_rgba))
                poses.append(original_pose @ trimesh_geom.graph.get(name)[0])
        else:
            # TODO: it is very weired that this condition works better for YCB texture
            exp_mesh = load_mesh(visual.filename, scale, trimesh_geom)
            geoms = [g.ObjMeshGeometry.from_stream(trimesh.util.wrap_as_stream(exp_mesh))]
            mats = [get_trimesh_geometry_material(trimesh_geom, default_rgba)]
            poses = [original_pose]
    else:
        raise NotImplementedError(f"Unrecognized type {visual.type}.")

    if visual.type != "File":
        rgb = np.clip(np.array(material.base_color[:3]) * 255, 0, 255).astype(np.uint8)
        mats = [
            g.MeshStandardMaterial(
                color=rgb_to_hex(rgb),
                opacity=float(material.base_color[3]),
                roughness=float(material.roughness),
                metalness=float(material.metallic),
            )
        ]
        poses = [pose]

    return geoms, mats, poses


def add_visual_to_viz(
    viz: meshcat.Visualizer, visual: sapien.VisualRecord, geom_root_path: str, geom_start_index: int
) -> int:
    pose = visual.pose.to_transformation_matrix().astype(float)
    geoms, mats, poses = visual2geom_mat(visual, pose)

    for geom, mat, pose in zip(geoms, mats, poses):
        geom_tree_path = f"{geom_root_path}/{geom_start_index}"
        viz[geom_tree_path].set_object(geom, mat)
        viz[geom_tree_path].set_transform(pose)
        geom_start_index += 1

    return geom_start_index


class MeshCatVisualizerSapien(MeshCatVisualizerBase):
    def __init__(self, port: Optional[int] = None, host="localhost", keep_default_viewer=True):
        super().__init__(port, host)
        self.keep_default_viewer = keep_default_viewer

        self.original_scene: Optional[sapien.Scene] = None
        self.new_scene: Optional[MimicScene] = None
        self.engine: Optional[sapien.Engine] = None
        self.render: Optional[sapien.IPxrRenderer] = None

        # Cache
        self.asset_resource_map: Dict[int, AssetResource] = {}
        self.asset_geom_map = {}
        self.env_map = {}
        self.actor_rigid_body_name_map: Dict[int, List[str]] = {}

    def set_scene(self, scene: sapien.Scene, engine: sapien.Engine, render: sapien.IPxrRenderer) -> sapien.Scene:
        # Default viewer setting
        self.viz["/Grid"].set_transform(np.eye(4))
        self.delete_all()

        self.original_scene = scene
        self.engine = engine
        self.render = render
        self.new_scene = MimicScene(self.original_scene)

        self._scene_override_builder_create_fn()
        self._scene_override_visualization_fn()
        self._scene_override_misc_fn()

        # noinspection PyTypeChecker
        return self.new_scene

    def _override_urdf_loader(self, mimic: MimicEntity):
        def add_urdf_visual(art: sapien.ArticulationBase, file_path: str, scale: float):
            filename = str((Path(file_path)).resolve())
            robot_id = art.get_links()[0].get_id()
            robot_tree_path = f"/Sim/articulation:{robot_id}"
            resource = self.dry_load_asset(filename, collapse_fixed_joints=False, replace_cylinder_with_capsule=False)
            self.load_asset_resources(resource, robot_tree_path, scale=scale)

        # Mimic class of sapien.URDFLoader
        def load(filename: str, config=None) -> sapien.Articulation:
            if config is None:
                config = {}
            robot = mimic.entity.load(filename, config)
            add_urdf_visual(robot, filename, mimic.scale)

            return robot

        def load_from_string(urdf_string: str, srdf_string: str, config=None) -> sapien.Articulation:
            if config is None:
                config = {}
            robot = mimic.entity.load_from_string(urdf_string, srdf_string, config)

            temp_urdf = tempfile.NamedTemporaryFile(suffix=".urdf")
            temp_urdf.write(urdf_string)
            add_urdf_visual(robot, temp_urdf.name, mimic.scale)

            return robot

        loader = self.original_scene.create_urdf_loader()

        def load_kinematic(filename: str, config=None) -> sapien.KinematicArticulation:
            if config is None:
                config = {}
            robot = mimic.entity.load_kinematic(filename, config)
            add_urdf_visual(robot, filename, mimic.scale)

            return robot

        def load_file_as_articulation_builder(filename: str, config=None) -> sapien.ArticulationBuilder:
            if config is None:
                config = {}
            builder = loader.load_file_as_articulation_builder(filename, config)
            mimic_builder = MimicEntity(builder)
            self._override_articulation_builder(mimic_builder)

            # noinspection PyTypeChecker
            return mimic_builder

        mimic.add_method("load", load)
        mimic.add_method("load_from_string", load_from_string)
        mimic.add_method("load_kinematic", load_kinematic)
        mimic.add_method("load_file_as_articulation_builder", load_file_as_articulation_builder)

    def _override_actor_builder(self, mimic: MimicEntity):
        def add_actor_visual(act: sapien.ActorBase):
            actor_tree_path = f"/Sim/actor:{act.get_id()}"
            geom_num = 0
            for i, visual in enumerate(mimic.get_visuals()):
                geom_num = add_visual_to_viz(self.viz, visual, actor_tree_path, geom_num)

        # Mimic class of sapien.ActorBuilder
        def build(name: str = "") -> sapien.Actor:
            actor = mimic.entity.build(name)
            add_actor_visual(actor)

            return actor

        def build_static(name: str = "") -> sapien.ActorStatic:
            actor = mimic.entity.build_static(name)
            add_actor_visual(actor)

            return actor

        def build_kinematic(name: str = "") -> sapien.Actor:
            actor = mimic.entity.build_kinematic(name)
            add_actor_visual(actor)

            return actor

        mimic.add_method("build", build)
        mimic.add_method("build_static", build_static)
        mimic.add_method("build_kinematic", build_kinematic)

    def _override_articulation_builder(self, mimic: MimicEntity):
        def add_robot_visual(art: sapien.ArticulationBase):
            robot_id = art.get_links()[0].get_id()
            robot_tree_path = f"/Sim/articulation:{robot_id}"
            for builder in mimic.entity.get_link_builders():
                link_name = builder.get_name() or f"actor-{builder.get_index()}"
                link_tree_path = f"{robot_tree_path}/{link_name}"
                geom_num = 0
                for i, visual in enumerate(builder.get_visuals()):
                    geom_num = add_visual_to_viz(self.viz, visual, link_tree_path, geom_num)

        def build(fix_root_link: bool = False) -> sapien.Articulation:
            robot = mimic.entity.build(fix_root_link)
            add_robot_visual(robot)

            return robot

        def build_kinematic() -> sapien.KinematicArticulation:
            robot = mimic.entity.build_kinematic()
            add_robot_visual(robot)

            return robot

        mimic.add_method("build", build)
        mimic.add_method("build_kinematic", build_kinematic)

    def _scene_override_builder_create_fn(self):
        def create_urdf_loader() -> sapien.URDFLoader:
            loader = self.original_scene.create_urdf_loader()
            mimic_loader = MimicEntity(loader)
            self._override_urdf_loader(mimic_loader)

            # noinspection PyTypeChecker
            return mimic_loader

        def create_actor_builder() -> sapien.ActorBuilder:
            builder = self.original_scene.create_actor_builder()
            mimic_builder = MimicEntity(builder)
            self._override_actor_builder(mimic_builder)

            # noinspection PyTypeChecker
            return mimic_builder

        def create_articulation_builder() -> sapien.ArticulationBuilder:
            builder = self.original_scene.create_articulation_builder()
            mimic_builder = MimicEntity(builder)
            self._override_articulation_builder(mimic_builder)

            # noinspection PyTypeChecker
            return mimic_builder

        self.new_scene.add_method("create_urdf_loader", create_urdf_loader)
        self.new_scene.add_method("create_actor_builder", create_actor_builder)
        self.new_scene.add_method("create_articulation_builder", create_articulation_builder)

    def _scene_override_misc_fn(self):
        def add_ground(
            altitude: float,
            render: bool = True,
            material: sapien.PhysicalMaterial = None,
            render_material: sapien.RenderMaterial = None,
            render_half_size: np.ndarray = np.array([10.0, 10.0], dtype=np.float32),
        ) -> sapien.ActorStatic:
            ground = self.original_scene.add_ground(altitude, render, material, render_material, render_half_size)
            if render:
                plane_size = render_half_size * 2
                plane = g.Plane(width=float(plane_size[0]), height=float(plane_size[1]))

                # Set default material for ground
                if render_material is None:
                    render_material = self.render.create_material()
                color = np.clip(render_material.base_color[:3] * 255, 0, 255).astype(np.uint8)
                material = g.MeshLambertMaterial(opacity=0.1, color=rgb_to_hex(color))
                self.viz["/Sim/ground"].set_object(plane, material)

                # Set ground pose
                scene_pose = np.eye(4)
                scene_pose[2, 3] = altitude
                self.viz["/Sim/ground"].set_transform(scene_pose)
                self.viz["/Grid"].set_transform(scene_pose)
            return ground

        self.new_scene.add_method("add_ground", add_ground)

    def _scene_override_visualization_fn(self):
        def update_render():
            self.original_scene.update_render()

            # Loop actor
            for actor in self.original_scene.get_all_actors():
                actor_tree_path = f"/Sim/actor:{actor.get_id()}"
                self.viz[actor_tree_path].set_transform(actor.get_pose().to_transformation_matrix().astype(float))

            for art in self.original_scene.get_all_articulations():
                robot_id = art.get_links()[0].get_id()
                robot_tree_path = f"/Sim/articulation:{robot_id}"
                for link in art.get_links():
                    link_tree_path = f"{robot_tree_path}/{link.get_name()}"
                    self.viz[link_tree_path].set_transform(link.get_pose().to_transformation_matrix().astype(float))

        self.new_scene.add_method("update_render", update_render)


_REGISTERED_VISUALIZER: List[MeshCatVisualizerSapien] = []


def create_sapien_visualizer(port=None, host="localhost", keep_default_viewer=True, **kwargs):
    visualizer = MeshCatVisualizerSapien(port, host, keep_default_viewer, **kwargs)
    if len(_REGISTERED_VISUALIZER) > 0:
        raise RuntimeError(f"You can only create a web visualizer once")
    visualizer.delete_all()
    _REGISTERED_VISUALIZER.append(visualizer)
    return visualizer


def bind_visualizer_to_sapien_scene(scene: sapien.Scene, engine: sapien.Engine, render: sapien.IPxrRenderer):
    if len(_REGISTERED_VISUALIZER) <= 0:
        raise RuntimeError(f"Web Visualizer has not been created yet! Call create_visualizer before register it to env")
    return _REGISTERED_VISUALIZER[0].set_scene(scene, engine, render)


def get_visualizer() -> MeshCatVisualizerSapien:
    if len(_REGISTERED_VISUALIZER) == 0:
        raise RuntimeError(f"No SAPIEN Web Visualizer is created.")
    return _REGISTERED_VISUALIZER[0]
