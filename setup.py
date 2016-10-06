# Setup script for megaman: scalable manifold learning
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import io
import os
import re
import sys
import subprocess

PY2 = sys.version_info[0] == 2
PY3 = not PY2
if PY3:
    import importlib.machinery


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py

    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'megaman'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('megaman')

    return config

DESCRIPTION = "megaman: Manifold Learning for Millions of Points"
LONG_DESCRIPTION = """
megaman: Manifold Learning for Millions of Points
=================================================

This repository contains a scalable implementation of several manifold learning
algorithms, making use of FLANN for fast approximate nearest neighbors and
PyAMG, LOBPCG, ARPACK, and other routines for fast matrix decompositions.

For more information, visit https://github.com/mmp2/megaman
"""
NAME = "megaman"
AUTHOR = "Marina Meila"
AUTHOR_EMAIL = "mmp@stat.washington.delete_this.edu"
URL = 'https://github.com/mmp2/megaman'
DOWNLOAD_URL = 'https://github.com/mmp2/megaman'
LICENSE = 'BSD 3'

VERSION = version('megaman/__init__.py')


def setup_package():
    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython()

    try:
        setup(name='megaman',
              author=AUTHOR,
              author_email=AUTHOR_EMAIL,
              url=URL,
              download_url=DOWNLOAD_URL,
              description=DESCRIPTION,
              long_description = LONG_DESCRIPTION,
              version=VERSION,
              license=LICENSE,
              configuration=configuration,
              classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Natural Language :: English',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5'])
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


if __name__ == '__main__':
    setup_package()
