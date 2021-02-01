from distutils.core import setup, Extension
import numpy

setup(name = 'cPSCS', version = '1.0',  \
   ext_modules = [Extension('clifford_t_estim', ['cmodule.c', 'AG.c', "QCircuit.c"], extra_compile_args=['-g'],include_dirs=[numpy.get_include()], libraries=["gsl", "gslcblas"])])
