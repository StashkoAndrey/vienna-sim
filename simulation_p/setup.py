from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "diffusion_cpp",
        ["diffusion.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="diffusion_cpp",
    version="0.1",
    ext_modules=ext_modules,
)