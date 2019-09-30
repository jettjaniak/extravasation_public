from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


sources = ["SimulationState.cpp", "Parameters.cpp", "forces.cpp", "helpers.cpp",
           "interpolated.cpp", "sampling.cpp", "velocities.cpp"]
sources = [f"../ekstrawazacja_cpp/{s}" for s in sources]

setup(ext_modules=cythonize(Extension(
    "simulation_engine",
    sources=["simulation_engine.pyx"] + sources,
    extra_compile_args=['-std=c++11'],
    include_dirs=["../ekstrawazacja_cpp/libs", numpy.get_include()],
    language="c++"
), annotate=True))