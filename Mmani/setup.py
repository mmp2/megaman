import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('Mmani', parent_package, top_path)

    config.add_subpackage('embedding')
    config.add_subpackage('embedding/tests')
    config.add_subpackage('geometry')
    config.add_subpackage('geometry/cyflann')
    config.add_subpackage('geometry/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())