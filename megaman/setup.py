# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('megaman', parent_package, top_path)

    config.add_subpackage('__check_build')
    config.add_subpackage('datasets')
    config.add_subpackage('embedding')
    config.add_subpackage('embedding/tests')
    config.add_subpackage('geometry')
    config.add_subpackage('geometry/cyflann')
    config.add_subpackage('geometry/tests')
    config.add_subpackage('plotter')
    config.add_subpackage('relaxation')
    config.add_subpackage('relaxation/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')
    config.add_data_files('geometry/tests/testmegaman_laplacian_rad0_2_lam1_5_n200.mat')
    config.add_data_files('relaxation/tests/eps_halfdome.mat')
    config.add_data_files('relaxation/tests/rloss_halfdome.mat')
    config.add_data_files('datasets/megaman.png')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
