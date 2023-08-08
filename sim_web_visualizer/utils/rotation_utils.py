# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import numpy as np
from meshcat import transformations as transformation


def compute_vector_rotation(origin_vector, target_vector):
    origin_vector /= max(np.linalg.norm(origin_vector), 1e-6)
    target_vector /= max(np.linalg.norm(target_vector), 1e-6)
    rotation_axis = np.cross(origin_vector, target_vector)
    if np.linalg.norm(rotation_axis) < 1e-6:
        return np.eye(4)
    cos = np.sum(origin_vector * target_vector)
    angle = np.arccos(cos)
    mat = transformation.rotation_matrix(angle, rotation_axis)
    return mat
