# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib
from sysconfig import get_paths

import importlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion


from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CONDA_PREFIX = os.environ['CONDA_PREFIX']

Extension = CUDAExtension
libraries = []
include_dirs = []
CC_FLAGS = []
NVCC_FLAGS = []

libraries.append('cudart')
CC_FLAGS += ["-O3"]
NVCC_FLAGS += ["-O3", "-Xcompiler=-fno-gnu-unique"]

extension = Extension(
    name='diffvg',
    sources=[
        'atomic.cpp',
        'color.cpp',
        'diffvg.cu',
        'parallel.cpp',
        'shape.cpp',
        'scene.cu'
    ],
    include_dirs=[
        os.path.join(CONDA_PREFIX, 'include')],
    extra_compile_args={'cxx': CC_FLAGS, 'nvcc': NVCC_FLAGS},
    extra_link_args=['-lcudart'],
    libraries=libraries
)


torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
build_with_cuda = True
if torch_spec is not None:
    packages.append('pydiffvg')
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
if tf_spec is not None and sys.platform != 'win32':
    packages.append('pydiffvg_tensorflow')
    if not build_with_cuda:
        import tensorflow as tf
        if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
            build_with_cuda = True
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed. For Windows platform only PyTorch is supported.')
    exit()


setup(name='diffvg',
      version='0.0.1',
      install_requires=["svgpathtools"],
      description='Differentiable Vector Graphics',
      ext_modules=[extension],
      cmdclass=dict(build_ext=BuildExtension.with_options(use_ninja=True), install=install),
      packages=packages,
      zip_safe=False)
