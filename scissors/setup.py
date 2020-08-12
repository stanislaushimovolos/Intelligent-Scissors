from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("search.pyx", sources=["scissors/search.pyx"],
              extra_compile_args=["-O3"], language="c++", annotate=True),
]

setup(
    name="processing_module",
    ext_modules=cythonize(extensions),
)
