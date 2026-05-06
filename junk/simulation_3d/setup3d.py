# --- setup3d.py ---
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "diffusion3d_cpp",
        ["diffusion3d.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17"]
    )
]

setup(
    name="diffusion3d_cpp",
    ext_modules=ext_modules
)
