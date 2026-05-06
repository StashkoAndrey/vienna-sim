from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# Correct path to your pybind11 headers
import pybind11
pybind11_include = pybind11.__file__

class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.include_dirs.append(pybind11_include)
        super().build_extensions()

ext_modules = [
    Extension(
        'diffusion_cpp',
        sources=['diffusion.cpp'],
        language='c++',
        extra_compile_args=['/std:c++17']
    ),
]

setup(
    name='diffusion_cpp',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
