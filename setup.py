from distutils.core import setup, Extension
import numpy

setup(name = 'clifford_t_estim', version = '1.0',  \
   ext_modules = [Extension('clifford_t_estim', ['cmodule.c', 'bitarray.c','CH.c','AG.c', "QCircuit.c"], extra_compile_args=["-O3"],include_dirs=[numpy.get_include()], libraries=["gsl", "gslcblas"])])
