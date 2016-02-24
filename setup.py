import io
import os
import re
import sys


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

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('Mmani')

    return config

DESCRIPTION = "Mmani: Scalable Manifold Learning"
LONG_DESCRIPTION = """
Mmani: Scalable Manifold Learning
=================================

This repository contains a scalable implementation of several manifold learning
algorithms, making use of FLANN for fast approximate nearest neighbors and
scipy routines for fast matrix decompositions.

For more information, visit https://github.com/mmp2/Mmani
"""
NAME = "Mmani"
AUTHOR = "Marina Meila"
URL = 'https://github.com/mmp2/Mmani'
DOWNLOAD_URL = 'https://github.com/mmp2/Mmani'
LICENSE = 'BSD 3'

VERSION = version('Mmani/__init__.py')


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

    try:
        setup(name='Mmani',
              author=AUTHOR,
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
                'Programming Language :: Python :: 2.7'],)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


if __name__ == '__main__':
    setup_package()
