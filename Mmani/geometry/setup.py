from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# use "export FLANN_ROOT=<FLANN_ROOT>"to set enviromental variable
# to build: python setup.py build_ext --inplace 

flann_path = os.environ['FLANN_ROOT']   

setup(ext_modules = cythonize(
    Extension(
           "cyflann_index",
           sources=["cyflann_index.pyx","flann_radius_neighbors.cc"],
           language="c++",
           extra_compile_args=["-O3", "-I" + flann_path + "/src/cpp/"],
    )))
