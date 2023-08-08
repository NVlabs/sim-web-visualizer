# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""Setup script """

import os

from setuptools import setup, find_packages


def collect_files(target_dir):
    file_list = []
    for (root, dirs, files) in os.walk(target_dir, followlinks=True):
        for filename in files:
            file_list.append(os.path.join('..', root, filename))
    return file_list


def setup_package():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    packages = find_packages(".")
    print(packages)

    package_files = ["assets/axis_geom/*"]

    setup(name='sim_web_visualizer',
      version='0.4.0',
      description='Web based visualizer for simulators',
      author='Yuzhe Qin',
      author_email='',
      url='',
      license='MIT',
      packages=packages,
      package_data={
          "sim_web_visualizer": package_files
      },
      python_requires='>=3.6,<3.11',
      install_requires=[
          # General dependencies
          "numpy<=1.23.0",
          "numpy-quaternion",
          "transforms3d",
          "dm_control>=1.0.0",
          "anytree",
          "trimesh",
          "pycollada",
          f"meshcat @ file://localhost{root_dir}/3rd_party/meshcat-python",
      ],
      )


setup_package()
