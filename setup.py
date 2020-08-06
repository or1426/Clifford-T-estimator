from distutils.core import setup, Extension
import numpy

setup(name = 'cPSCS', version = '1.0',  \
   ext_modules = [Extension('cPSCS', ['cmodule.c'], extra_compile_args=['-g'],include_dirs=[numpy.get_include()])])
