from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


sources = ["SimulationState.cpp", "Parameters.cpp", "forces.cpp", "helpers.cpp",
           "interpolated.cpp", "sampling.cpp", "velocities.cpp"]
sources = [f"../ekstrawazacja_cpp/{s}" for s in sources]

setup(ext_modules=cythonize(Extension(
    "simulation_engine",
    extra_compile_args = ["-std=c++11", "-mmacosx-version-min=10.9","-Wno-c++11-narrowing"],
    extra_link_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"],
    sources=["simulation_engine.pyx"] + sources,
    include_dirs=["/usr/local/include/eigen3","../ekstrawazacja_cpp/libs", numpy.get_include()],
    language="c++"
), annotate=True))
