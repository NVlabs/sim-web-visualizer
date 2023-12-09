# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""Setup script """

import re
from pathlib import Path

from setuptools import setup, find_packages

_here = Path(__file__).resolve().parent
name = "sim_web_visualizer"

# Reference: https://github.com/kevinzakka/mjc_viewer/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

core_requirements = [
    "numpy<=1.23.0",
    "numpy-quaternion",
    "transforms3d",
    "dm_control>=1.0.0",
    "anytree",
    "trimesh",
    "pycollada",
    "mujoco>=2.2.0",
    "meshcat-sim-web-fork",
]

isaac_requirements = [
    "hydra-core",
    "gym==0.23.1",
    "rl-games",
    "torch",
    "pyvirtualdisplay",
    "omegaconf",
    "jinja2",
]

sapien_requirements = [
    "mani-skill2",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]


def setup_package():
    # Meta information of the project
    author = "Yuzhe Qin"
    author_email = "y1qin@ucsd.edu"
    description = "Web based visualizer for simulators"
    url = "https://github.com/NVlabs/sim-web-visualizer"
    with open(_here / "README.md", "r") as file:
        readme = file.read()

    # Package data
    packages = find_packages(".")
    print(f"Packages: {packages}")
    package_files = ["assets/axis_geom/*"]

    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        maintainer=author,
        maintainer_email=author_email,
        description=description,
        long_description=readme,
        long_description_content_type="text/markdown",
        url=url,
        license="MIT",
        license_files=("LICENSE",),
        packages=packages,
        package_data={name: package_files},
        python_requires=">=3.6,<3.11",
        zip_safe=True,
        install_requires=core_requirements,
        extras_require={
            "isaacgym": isaac_requirements,
            "sapien": sapien_requirements,
        },
        classifiers=classifiers,
    )


setup_package()
