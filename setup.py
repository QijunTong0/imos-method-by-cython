from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="imos_cython",
    version="0.1",
    ext_modules=cythonize("imos_cython.pyx"),
    include_dirs=[numpy.get_include()],
)
# >>> python setup.py build_ext --inplace
