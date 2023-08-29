# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import io
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import meshcat.geometry as g
import numpy as np
import trimesh


@dataclass
class AssetResource:
    filename: str
    visual_data: Dict[str, Tuple[g.Geometry, g.Material]]
    pose_data: Dict[str, np.ndarray]


def load_mesh(mesh_file: str, scale=np.array([1, 1, 1]), mesh: Optional[trimesh.Trimesh] = None):
    # Reference: https://github.com/petrikvladimir/RoboMeshCat/blob/main/src/robomeshcat/object.py
    if mesh is None:
        try:
            mesh: trimesh.Trimesh = trimesh.load(mesh_file, force="mesh")
        except ValueError as e:
            if str(e) == "File type: dae not supported":
                print(
                    "To load DAE meshes you need to install pycollada package via "
                    "`conda install -c conda-forge pycollada`"
                    " or `pip install pycollada`"
                )
            raise
        except Exception as e:
            print(
                f"Loading of a mesh failed with message: '{e}'. "
                f"Trying to load with with 'ignore_broken' but consider to fix the mesh located here:"
                f" '{mesh_file}'."
            )
            mesh: trimesh.Trimesh = trimesh.load(mesh_file, force="mesh", ignore_broken=True)

    mesh.apply_scale(scale)
    try:
        if mesh_file.lower().endswith("dae"):
            exp_mesh = trimesh.exchange.dae.export_collada(mesh)
        else:
            exp_mesh = trimesh.exchange.obj.export_obj(mesh)
    except ValueError:
        if mesh_file.lower().endswith("dae"):
            exp_mesh = trimesh.exchange.dae.export_collada(mesh, include_texture=False)
        else:
            exp_mesh = trimesh.exchange.obj.export_obj(mesh, include_texture=False)

    return exp_mesh


def get_trimesh_geometry_material(geom: trimesh.Trimesh, default_rgba: Optional[np.ndarray] = None) -> g.Material:
    use_urdf_color = default_rgba is not None
    if use_urdf_color:
        default_rgba = (np.clip(default_rgba, 0, 1) * 255).astype(int)
    if isinstance(geom.visual, trimesh.visual.texture.TextureVisuals):
        material = geom.visual.material

        if isinstance(material, trimesh.visual.material.SimpleMaterial):
            image = material.image
            mat = g.MeshPhongMaterial(shininess=material.glossiness, specular=rgb_to_hex(material.specular[:3]), side=0)
            if image is not None and np.prod(image.size) > 100:
                output = io.BytesIO()
                image.save(output, format="png")
                hex_data = output.getvalue()
                texture = g.ImageTexture(g.PngImage(data=hex_data))
                mat.map = texture
            else:
                rgba = default_rgba if use_urdf_color else material.main_color
                mat.color = rgb_to_hex(rgba[:3])
                mat.opacity = rgba[3] / 255.0
            return mat
        elif isinstance(material, trimesh.visual.material.PBRMaterial):
            mat = g.MeshStandardMaterial(metalness=material.metallicFactor, roughness=material.roughnessFactor)
            if not material.doubleSided:
                mat.side = 0
            if material.baseColorTexture is not None:
                output = io.BytesIO()
                material.baseColorTexture.save(output, format=material.baseColorTexture.format)
                hex_data = output.getvalue()
                texture = g.ImageTexture(image=g.PngImage(data=hex_data))
                mat.map = texture
            else:
                rgba = default_rgba if use_urdf_color else material.main_color
                mat.color = rgb_to_hex(rgba[:3])
                mat.opacity = rgba[3] / 255.0
            return mat
        else:
            raise NotImplementedError(f"Type f{type(material)} not supported.")

    elif isinstance(geom.visual, trimesh.visual.color.ColorVisuals):
        if use_urdf_color:
            rgba = default_rgba
        else:
            rgba = geom.visual.main_color
        return g.MeshLambertMaterial(color=rgb_to_hex(rgba[:3]), opacity=rgba[3] / 255.0)


def rgb_to_hex(rgb):
    return "0x{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
