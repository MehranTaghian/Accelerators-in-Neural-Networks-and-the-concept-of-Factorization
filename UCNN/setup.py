import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=(
    cythonize(Extension("Factorization", ["Factorization.pyx"]))),
    include_dirs=[numpy.get_include()],
    requires=['numpy', 'Cython'])

setup(ext_modules=(
    cythonize(Extension("UCNN_lane", ["UCNN_lane.pyx"]))),
    include_dirs=[numpy.get_include()],
    requires=['numpy', 'Cython'])

setup(ext_modules=(
    cythonize(Extension("Convolve", ["Convolve.pyx"]))),
    include_dirs=[numpy.get_include()],
    requires=['numpy', 'Cython'])
