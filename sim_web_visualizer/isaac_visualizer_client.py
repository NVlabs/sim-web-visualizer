# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

import meshcat
import meshcat.geometry as g
import numpy as np
import quaternion
import torch
import transforms3d
from isaacgym import gymapi
from isaacgym import gymtorch
from numpy.lib.recfunctions import structured_to_unstructured

from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase, AssetResource
from sim_web_visualizer.utils.rotation_utils import compute_vector_rotation

USE_GPU_PIPELINE = False


class MimicGym:
    def __init__(self, gym: gymapi.Gym):
        self._gym = gym
        self.methods: Dict[str, Callable] = {}

    def add_method(self, name: str, fn: Callable):
        self.methods[name] = fn

    def __getattribute__(self, item):
        if item in ["add_method", "methods", "_gym"]:
            return super().__getattribute__(item)
        if item in self.methods:
            return self.methods[item]
        else:
            return self._gym.__getattribute__(item)


class MimicViewer:
    def __init__(self):
        pass


def set_env_pose(
    num_env: int, num_per_row: int, env_size: np.ndarray, scene_offset: np.ndarray, viz: meshcat.Visualizer
):
    row = num_env // num_per_row
    column = num_env % num_per_row
    env_origin_xy = env_size[:2] * np.array([column, row])
    env_pose = np.eye(4)
    env_pose[:2, 3] = env_origin_xy - scene_offset
    viz[f"/Sim/env:{num_env}"].set_transform(env_pose)


class MeshCatVisualizerIsaac(MeshCatVisualizerBase):
    def __init__(
        self,
        port: Optional[int] = None,
        host="localhost",
        keep_default_viewer=True,
        scene_offset=np.array([10.0, 10.0]),
        max_env=4,
    ):
        super().__init__(port, host)
        self.keep_default_viewer = keep_default_viewer

        self.original_gym: Optional[gymapi.Gym] = None
        self.new_gym: Optional[MimicGym] = None
        self.sim = None
        self.max_env = max_env
        print(
            f"For efficiency, the sim_web_visualizer will only visualize the first {max_env} environments.\n"
            f"Modify the max_env parameter if you prefer more or less visualization."
        )

        # Cache
        self.asset_resource_map: Dict[int, AssetResource] = {}
        self.asset_geom_map = {}
        self.actor_asset_map = {}
        self.env_map = {}
        self.actor_rigid_body_name_map: Dict[int, List[str]] = {}

        # GPU pipeline settings
        self.prepared = False
        self.rigid_body_state_tensor: Optional[torch.Tensor] = None

        self.env_handle_request = 0
        self.num_per_row = None
        self.env_size = None
        self.env_actor_count = {}
        self.scene_tree = {}

        # Triangle mesh
        self.triangle_mesh_count = 0

        # Viewer cache
        self.viewer_keyboard_event = {}
        self.viewer_mouse_event = {}

        # Global setting
        self.scene_offset = scene_offset
        scene_pose = np.eye(4)
        scene_pose[:2, 3] = self.scene_offset
        self.viz["/Scene/Axes"].set_transform(scene_pose)

    def set_gym_instance(self, gym: gymapi.Gym, sim: gymapi.Sim) -> gymapi.Gym:
        self.original_gym = gym
        self.sim = sim
        self.new_gym = MimicGym(self.original_gym)

        self._override_load_asset()
        self._override_asset_create_fn()

        self._overload_create_env()
        self._override_create_actor()

        self._overload_end_aggregate()
        self._override_set_fn()
        self._override_viewer_fn()

        self._override_add_fn()
        # noinspection PyTypeChecker
        return self.new_gym

    def _override_load_asset(self):
        def load_asset(
            sim: gymapi.Sim, rootpath: str, filename: str, options: gymapi.AssetOptions = gymapi.AssetOptions()
        ):
            asset = self.original_gym.load_asset(sim, rootpath, filename, options)

            filename = str((Path(rootpath) / filename).resolve())
            collapse_fixed_joints = options.collapse_fixed_joints

            # Load asset into the memory as asset resource but not forward it into the web viewer
            # The asset will only be loaded into web viewer after a `create_actor` call
            resource = self.dry_load_asset(
                filename,
                collapse_fixed_joints,
                replace_cylinder_with_capsule=options.replace_cylinder_with_capsule,
                use_mesh_materials=options.use_mesh_materials,
            )
            self.asset_resource_map[id(asset)] = resource
            return asset

        self.new_gym.add_method("load_asset", load_asset)

    def _override_asset_create_fn(self):
        def create_box(
            sim: gymapi.Sim, width: float, height: float, depth: float, options: gymapi.AssetOptions = ...
        ) -> gymapi.Asset:
            asset = self.original_gym.create_box(sim, width, height, depth, options)

            geom = g.Box([width, height, depth])
            self.asset_geom_map[id(asset)] = geom
            return asset

        def create_sphere(sim: gymapi.Sim, radius: float, options: gymapi.AssetOptions) -> gymapi.Asset:
            asset = self.original_gym.create_sphere(sim, radius, options)

            geom = g.Sphere(radius)
            self.asset_geom_map[id(asset)] = geom
            return asset

        self.new_gym.add_method("create_box", create_box)
        self.new_gym.add_method("create_sphere", create_sphere)

    def _overload_create_env(self):
        def create_env(arg0: gymapi.Sim, arg1: gymapi.Vec3, arg2: gymapi.Vec3, arg3: int) -> gymapi.Env:
            env = self.original_gym.create_env(arg0, arg1, arg2, arg3)
            num_env_now = self.original_gym.get_env_count(self.sim)
            if num_env_now <= self.max_env:
                self.env_map[env] = num_env_now - 1
                if self.env_handle_request > 0:
                    num_env = num_env_now - 1 - self.env_handle_request
                    set_env_pose(num_env, self.num_per_row, self.env_size, np.zeros(2), self.viz)
                    self.env_handle_request -= 1

                self.env_handle_request += 1

                if self.num_per_row is None:
                    self.num_per_row = min(arg3, int(np.ceil(np.sqrt(self.max_env))))
                    env_size = arg2 - arg1
                    self.env_size = np.array([env_size.x, env_size.y, env_size.z])
            return env

        self.new_gym.add_method("create_env", create_env)

    def _overload_end_aggregate(self):
        def end_aggregate(arg0: gymapi.Env) -> bool:
            result = self.original_gym.end_aggregate(arg0)

            return result

        self.new_gym.add_method("end_aggregate", end_aggregate)

    def _override_create_actor(self):
        def create_actor(
            env: gymapi.Env,
            asset: gymapi.Asset,
            pose: gymapi.Transform,
            name: str = "",
            collision_group: int = 0,
            collision_filter: int = 0,
            seg_id: int = 0,
        ) -> int:
            actor = self.original_gym.create_actor(env, asset, pose, name, collision_group, collision_filter, seg_id)

            if env in self.env_map:
                num_env = self.env_map[env]
                if num_env in self.env_actor_count:
                    self.env_actor_count[num_env] += 1
                else:
                    self.env_actor_count[num_env] = 1
                num_actor = self.env_actor_count[num_env]
                robot_tree_path = f"/Sim/env:{num_env}/actor:{num_actor - 1}"

                if id(asset) in self.asset_resource_map:
                    resource = self.asset_resource_map[id(asset)]
                    self.load_asset_resources(resource, robot_tree_path, scale=1.0)
                else:
                    geom = self.asset_geom_map[id(asset)]
                    rigid_body_name = self.original_gym.get_actor_rigid_body_names(env, actor)[0]
                    geom_path = f"{robot_tree_path}/{rigid_body_name}"
                    self.viz[geom_path].set_object(geom)

                self.actor_asset_map[f"{num_env}/{num_actor-1}"] = id(asset)

            return actor

        self.new_gym.add_method("create_actor", create_actor)

    def _override_add_fn(self):
        def add_ground(sim: gymapi.Sim, plane_params: gymapi.PlaneParams):
            self.original_gym.add_ground(sim, plane_params)

            # Compute the pose of ground
            distance = plane_params.distance
            normal = plane_params.normal
            normal = normal / normal.length()
            normal = np.array([normal.x, normal.y, normal.z])
            pos = normal * distance
            rotation = compute_vector_rotation(np.array([0, 0, 1.0]), normal)
            pose = np.eye(4)
            pose[:3, :3] = rotation[:3, :3]
            pose[:3, 3] = pos

            plane_size = self.scene_offset * 2 + 5
            plane = g.Plane(width=plane_size[0], height=plane_size[1])

            # Set default material for ground
            material = g.MeshLambertMaterial(opacity=0.1, color="0x8E8F94")
            self.viz["/Sim/ground"].set_object(plane, material)

            # Set ground pose
            grid_scale = max(self.scene_offset) / 10.0
            scene_pose = np.eye(4)
            scene_pose[:2, 3] = self.scene_offset
            self.viz["/Sim/ground"].set_transform(scene_pose)
            scene_pose[:3, :3] *= grid_scale
            self.viz["/Grid"].set_transform(scene_pose)

        def add_triangle_mesh(
            sim: gymapi.Sim, vertices: np.ndarray, triangles: np.ndarray, params: gymapi.TriangleMeshParams
        ):
            self.original_gym.add_triangle_mesh(sim, vertices, triangles, params)

            # Create meshcat mesh
            v = vertices.reshape([-1, 3])
            f = triangles.reshape([-1, 3])
            geom = g.TriangularMeshGeometry(v, f)
            self.viz[f"/Sim/triangle_mesh/{self.triangle_mesh_count}"].set_object(geom)

            # Set mesh pose
            transform = params.transform
            pose = np.eye(4)
            pose[:3, 3] = [transform.p.x, transform.p.y, transform.p.z]
            quat = [transform.r.w, transform.r.x, transform.r.y, transform.r.z]
            pose[:3, :3] = transforms3d.quaternions.quat2mat(quat)
            self.viz[f"/Sim/triangle_mesh/{self.triangle_mesh_count}"].set_transform(pose)
            self.triangle_mesh_count += 1

        self.new_gym.add_method("add_ground", add_ground)
        self.new_gym.add_method("add_triangle_mesh", add_triangle_mesh)

    def _override_set_fn(self):
        def set_rigid_body_color(arg0: gymapi.Env, arg1: int, arg2: int, arg3: gymapi.MeshType, arg4: gymapi.Vec3):
            self.original_gym.set_rigid_body_color(arg0, arg1, arg2, arg3, arg4)
            if arg0 in self.env_map:
                body_name = self.original_gym.get_actor_rigid_body_names(arg0, arg1)[arg2]

                if arg3 in [gymapi.MESH_VISUAL, gymapi.MESH_VISUAL_AND_COLLISION]:
                    env_num = self.env_map[arg0]
                    body_path = f"/Sim/env:{env_num}/actor:{arg1}/{body_name}"
                    rgba = [arg4.x, arg4.y, arg4.z, 1]
                    self.viz[body_path].set_property("color", rgba)

        def set_actor_scale(arg0: gymapi.Env, actor_id: int, scale: float):
            self.original_gym.set_actor_scale(arg0, actor_id, scale)
            if arg0 in self.env_map:
                env_num = self.env_map[arg0]
                actor_path = f"/Sim/env:{env_num}/actor:{actor_id}"
                asset_id = self.actor_asset_map[f"{env_num}/{actor_id}"]
                resource = self.asset_resource_map[asset_id]

                for path, data in resource.visual_data.items():
                    pose = resource.pose_data[path]
                    scaling_pose = pose.copy()
                    scaling_pose[:3] *= scale
                    self.viz[f"/{actor_path}/{path}"].set_transform(scaling_pose)

        self.new_gym.add_method("set_rigid_body_color", set_rigid_body_color)
        self.new_gym.add_method("set_actor_scale", set_actor_scale)

    def _override_viewer_fn(self):
        def create_viewer(sim: gymapi.Sim, cam_prop: gymapi.CameraProperties) -> Union[MimicViewer, gymapi.Viewer]:
            if self.keep_default_viewer:
                viewer = self.original_gym.create_viewer(sim, cam_prop)
                return viewer
            else:
                return MimicViewer()

        def subscribe_viewer_keyboard_event(
            viewer: Union[MimicViewer, gymapi.Viewer], keyboard_input: gymapi.KeyboardInput, event_name: str
        ):
            if self.keep_default_viewer:
                self.original_gym.subscribe_viewer_keyboard_event(viewer, keyboard_input, event_name)
            self.viewer_keyboard_event[keyboard_input.name] = event_name

        def subscribe_viewer_mouse_event(
            viewer: Union[MimicViewer, gymapi.Viewer], mouse_input: gymapi.MouseInput, event_name: str
        ):
            if self.keep_default_viewer:
                self.original_gym.subscribe_viewer_mouse_event(viewer, mouse_input, event_name)
            self.viewer_mouse_event[mouse_input.name] = event_name

        def viewer_camera_look_at(
            viewer: Union[MimicViewer, gymapi.Viewer], env: gymapi.Env, pos: gymapi.Vec3, target: gymapi.Vec3
        ):
            if self.keep_default_viewer:
                self.original_gym.viewer_camera_look_at(viewer, env, pos, target)
            # TODO:
            pass

        def query_viewer_has_closed(viewer: Union[MimicViewer, gymapi.Viewer]) -> bool:
            if self.keep_default_viewer:
                self.original_gym.query_viewer_has_closed(viewer)
            return False

        def query_viewer_action_events(viewer: Union[MimicViewer, gymapi.Viewer]) -> List[gymapi.ActionEvent]:
            if self.keep_default_viewer:
                action = self.original_gym.query_viewer_action_events(viewer)
            else:
                # TODO:
                action_class = namedtuple("ActionEvent", ["action", "value"])
                action = [action_class(action="", value=0.0)]
            return action

        def draw_viewer(viewer: Union[MimicViewer, gymapi.Viewer], sim: gymapi.Sim, render_collision: bool):
            sim_viz = self.viz["/Sim"]
            inverse_env_map = {k: v for v, k in self.env_map.items()}

            if self.keep_default_viewer:
                self.original_gym.draw_viewer(viewer, sim, render_collision)

            if self.env_handle_request > 0:
                num_env = min(self.original_gym.get_env_count(self.sim), self.max_env) - self.env_handle_request
                set_env_pose(num_env, self.num_per_row, self.env_size, np.zeros(2), self.viz)
                self.env_handle_request -= 1

            if USE_GPU_PIPELINE and not self.prepared:
                num_envs = self.original_gym.get_env_count(self.sim)
                rigid_body_tensor = self.original_gym.acquire_rigid_body_state_tensor(self.sim)
                self.original_gym.refresh_rigid_body_state_tensor(self.sim)
                self.rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(num_envs, -1, 13)
                print(f"Setup tensor for GPU pipeline.")

            if not self.prepared:
                env_ptr = inverse_env_map[0]
                for k in range(self.env_actor_count[0]):
                    rigid_body_name = self.original_gym.get_actor_rigid_body_names(env_ptr, k)
                    self.actor_rigid_body_name_map[k] = rigid_body_name
                self.prepared = True

            if USE_GPU_PIPELINE:
                self.original_gym.refresh_rigid_body_state_tensor(self.sim)
                rigid_body_state = self.rigid_body_state_tensor[: len(self.env_map), :, :7].cpu().numpy()
                pos = rigid_body_state[:, :, :3]
                quat = np.concatenate([rigid_body_state[..., 6:7], rigid_body_state[..., 3:6]], axis=-1)  # xyzw -> wxyz
                qs = quaternion.as_quat_array(quat)
                rot = quaternion.as_rotation_matrix(qs)

                pose = np.tile(np.eye(4), rigid_body_state.shape[0:2] + (1, 1))
                pose[..., :3, :3] = rot
                pose[..., :3, 3] = pos

                for i in range(len(self.env_map)):
                    env_viz = sim_viz[f"env:{i}"]
                    body_count = 0
                    for k in range(self.env_actor_count[i]):
                        actor_viz = env_viz[f"actor:{k}"]
                        rigid_body_name = self.actor_rigid_body_name_map[k]
                        rigid_body_count = len(rigid_body_name)
                        for rigid_body_idx in range(body_count, body_count + rigid_body_count):
                            actor_viz[rigid_body_name[rigid_body_idx - body_count]].set_transform(
                                pose[i, rigid_body_idx]
                            )
                        body_count += rigid_body_count
            else:
                for i in range(len(self.env_map)):
                    env_ptr = inverse_env_map[i]
                    env_viz = sim_viz[f"env:{i}"]
                    for k in range(self.env_actor_count[i]):
                        actor_viz = env_viz[f"actor:{k}"]
                        actor_states = self.original_gym.get_actor_rigid_body_states(env_ptr, k, gymapi.STATE_POS)
                        rigid_body_name = self.actor_rigid_body_name_map[k]
                        rigid_body_count = len(rigid_body_name)
                        r = actor_states["pose"]["r"]
                        p = actor_states["pose"]["p"]
                        wxyz = structured_to_unstructured(r[["w", "x", "y", "z"]])
                        qs = quaternion.as_quat_array(wxyz)

                        rot = quaternion.as_rotation_matrix(qs)
                        pos = structured_to_unstructured(p[["x", "y", "z"]])
                        pose = np.tile(np.eye(4), (rigid_body_count, 1, 1))
                        pose[:, :3, :3] = rot
                        pose[:, :3, 3] = pos

                        for rigid_body_idx in range(rigid_body_count):
                            actor_viz[rigid_body_name[rigid_body_idx]].set_transform(pose[rigid_body_idx])

        self.new_gym.add_method("create_viewer", create_viewer)
        self.new_gym.add_method("subscribe_viewer_keyboard_event", subscribe_viewer_keyboard_event)
        self.new_gym.add_method("subscribe_viewer_mouse_event", subscribe_viewer_mouse_event)
        self.new_gym.add_method("viewer_camera_look_at", viewer_camera_look_at)
        self.new_gym.add_method("query_viewer_has_closed", query_viewer_has_closed)
        self.new_gym.add_method("query_viewer_action_events", query_viewer_action_events)

        # Main step function: draw_viewer
        self.new_gym.add_method("draw_viewer", draw_viewer)
        self.new_gym.add_method("poll_viewer_events", partial(draw_viewer, sim=self.sim, render_collision=False))


_REGISTERED_VISUALIZER: List[MeshCatVisualizerIsaac] = []


def create_isaac_visualizer(port=None, host="localhost", keep_default_viewer=True, max_env=4, **kwargs):
    visualizer = MeshCatVisualizerIsaac(port, host, keep_default_viewer, max_env=max_env, **kwargs)
    if len(_REGISTERED_VISUALIZER) > 0:
        raise RuntimeError(f"You can only create a web visualizer once")
    visualizer.delete_all()
    _REGISTERED_VISUALIZER.append(visualizer)
    return visualizer


def bind_visualizer_to_gym(gym: gymapi.Gym, sim: gymapi.Sim):
    if len(_REGISTERED_VISUALIZER) <= 0:
        raise RuntimeError(f"Web Visualizer has not been created yet! Call create_visualizer before register it to env")
    return _REGISTERED_VISUALIZER[0].set_gym_instance(gym, sim)


def set_gpu_pipeline(use_gpu_pipeline: bool):
    global USE_GPU_PIPELINE
    USE_GPU_PIPELINE = use_gpu_pipeline


def get_visualizer() -> MeshCatVisualizerIsaac:
    if len(_REGISTERED_VISUALIZER) == 0:
        raise RuntimeError(f"No IsaacGym Web Visualizer is created.")
    return _REGISTERED_VISUALIZER[0]
