#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    # Everything but primes.pyx is included here.
    Extension("*", ["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])
]
setup(
    name="Fast outer product and sum",
    ext_modules=cythonize(extensions),
)