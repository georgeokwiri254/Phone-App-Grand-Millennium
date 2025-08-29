from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "revenue_analytics_cpp",
        ["cpp_module/revenue_analytics.cpp"],
        cxx_std=14,  # C++14 standard
        include_dirs=[
            pybind11.get_include(),
        ],
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="revenue_analytics_cpp",
    version="0.1.0",
    description="Fast C++ revenue analytics computations for Streamlit",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.6.0",
    ],
)