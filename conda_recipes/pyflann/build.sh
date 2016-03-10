#!/bin/bash

cd src/python
cmake . -DLIBRARY_OUTPUT_PATH=$PREFIX/lib -DFLANN_VERSION="$PKG_VERSION"
$PYTHON setup.py install
