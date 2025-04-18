from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="imos_cython",
    ext_modules=cythonize(
        Extension(
            name="imos_cython",
            sources=["imos_cython.pyx"],
            extra_compile_args=["-O2"],
        )
    ),
    include_dirs=[numpy.get_include()],
)
# >>> python setup.py build_ext --inplace
