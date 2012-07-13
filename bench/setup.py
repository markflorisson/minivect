from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import numpy

setup(
    ext_modules=[
        Extension('bench', ['bench.pyx'], include_dirs=[numpy.get_include()],
                #extra_link_args=['-fopenmp'], extra_compile_args=['-fopenmp'],
                extra_objects=['fbench.o'],
        ),
    ],
    cmdclass={'build_ext': build_ext},
)


