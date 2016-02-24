import os
import sys


flann_root = os.environ.get('FLANN_ROOT', sys.exec_prefix)
print("Compiling FLANN with FLANN_ROOT={0}".format(flann_root))

flann_include = os.path.join(flann_root, 'include')
flann_lib = os.path.join(flann_root, 'lib')


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('geometry/cyflann', parent_package, top_path)
    libraries = ['flann', 'flann_cpp']
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("index",
           sources=["index.cpp", "cyflann_index.cc"],
           include_dirs=[numpy.get_include(), flann_include],
           libraries = libraries,
           library_dirs = [flann_lib],
           runtime_library_dirs= [flann_lib],
           extra_compile_args=["-O3"])

    return config
