# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import warnings

from .base_visualizer_client import MeshCatVisualizerBase

__version__ = "0.6.0"

try:
    import sapien.core as sapien
    from .sapien_visualizer_client import create_sapien_visualizer, bind_visualizer_to_sapien_scene, get_visualizer
except ImportError as e:
    warnings.warn(str(e))
    warnings.warn(
        f"\nNo Sapien python library installed. Disable Sapien Visualizer.\n "
        f"If you want to Sapien Visualizer, please consider install it via: pip3 install sapien"
    )

try:
    from isaacgym import gymapi
    from .isaac_visualizer_client import set_gpu_pipeline, create_isaac_visualizer, bind_visualizer_to_gym
except ImportError as e:
    warnings.warn(str(e))
    warnings.warn(
        f"\nNo isaacgym python library installed. Disable IsaacGym Visualizer.\n"
        f"If you want to IsaacGym Visualizer, please consider install it via the following URL: "
        f"https://developer.nvidia.com/isaac-gym"
    )
