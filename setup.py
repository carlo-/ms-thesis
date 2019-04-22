import os
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='DTW in Cython',
    ext_modules=cythonize('thesis_tools/dtw/dtw_c.pyx')
)

os.remove('thesis_tools/dtw/dtw_c.c')

# Build with:
# python setup.py build_ext --inplace
