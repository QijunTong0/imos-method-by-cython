from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="cython_experiment",
    ext_modules=cythonize(
        Extension(
            name="cython_module",
            sources=["src_cython.pyx"],
            extra_compile_args=["-O2"],
        )
    ),
    include_dirs=[numpy.get_include()],
)
# >>> python setup.py build_ext --inplace
