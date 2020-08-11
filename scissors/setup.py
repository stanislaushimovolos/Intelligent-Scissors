from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("processing_module", sources=["search.pyx"], include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"], language="c++", annotate=True),
]

setup(
    name="processing_module",
    ext_modules=cythonize(extensions),
)
