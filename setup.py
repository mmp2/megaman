import io
import os
import re

from distutils.core import setup


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

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['Mmani',
                'Mmani.embedding',
                'Mmani.embedding.tests',
                'Mmani.geometry',
                'Mmani.geometry.tests',
                'Mmani.utils',
                'Mmani.utils.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7'],
     )
