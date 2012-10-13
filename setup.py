import os
import sys
from fnmatch import fnmatchcase
from distutils.util import convert_path
from distutils.core import setup

kwds = {}
kwds['long_description'] = open('README.md').read()

setup(
    name = "minivect",
    author = "Mark Florisson",
    author_email = "mark.florisson@continuum.io",
    url = "https://github.com/markflorisson88/minivect",
    license = "BSD",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.4",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Topic :: Software Development :: Compilers",
    ],
    description = "Compiler backend for array expressions",
    packages = ['minivect', 'minivect.pydot', 'minivect.tests'],
    package_data = {
        'minivect' : ['include/*'],
    },
    version = '0.1',
)
