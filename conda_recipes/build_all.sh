conda config --set anaconda_upload yes
conda build flann
conda build --py all pyflann
conda build --python 2.7 --python 3.4 --python 3.5 --numpy 1.9 --numpy 1.10 pyamg
conda build --python 2.7 --python 3.4 --python 3.5 --numpy 1.10 megaman
