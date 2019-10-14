from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=(cythonize(Extension("factorization", ["Accelerator_cython\\factorization.pyx"]))),
      include_dirs=[numpy.get_include()],
      requires=['numpy', 'Cython', 'tensorflow', 'xlsxwriter', 'open'])
