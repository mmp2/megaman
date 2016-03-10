# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import os
import sys
import platform

FLANN_ROOT = os.environ.get('FLANN_ROOT', sys.exec_prefix)
CONDA_BUILD = os.environ.get('CONDA_BUILD', 0)

def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('geometry/cyflann', parent_package, top_path)
    libraries = ['flann', 'flann_cpp']
    if os.name == 'posix':
        libraries.append('m')

    kwds = {}
    flann_include = os.path.join(FLANN_ROOT, 'include')
    flann_lib = os.path.join(FLANN_ROOT, 'lib')

    if CONDA_BUILD:
        # conda uses relative dynamic library paths
        pass
    else:
        # direct installations use absolute library paths
        print("Compiling FLANN with FLANN_ROOT={0}".format(FLANN_ROOT))

        # from http://stackoverflow.com/questions/19123623/python-runtime-library-dirs-doesnt-work-on-mac
        if platform.system() == 'Darwin':
            kwds['extra_link_args'] = ['-Wl,-rpath,'+flann_lib]
        kwds['runtime_library_dirs'] = [flann_lib]

    config.add_extension("index",
           sources=["index.cxx", "cyflann_index.cc"],
           include_dirs=[numpy.get_include(), flann_include],
           libraries=libraries,
           library_dirs=[flann_lib],
           extra_compile_args=["-O3"],
           **kwds)

    return config
