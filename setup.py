import sys
from setuptools import setup, Extension, find_packages
import numpy
try:
    from Cython.Build import build_ext, cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    Cython.Compiler.Options.docstrings = True
    use_Cython = True
except:
    use_Cython = False

if use_Cython:
    ext_modules = cythonize([
        Extension(
            "VitLib_cython",
            ["src/VitLib_cython.pyx"],
            language="c++",
        )
    ])
else:
    ext_modules = [
        Extension(
            "VitLib_cython",
            ["src/VitLib_cython.cpp"],
            language="c++",
        )
    ]

setup_kwargs = {
    "name": "VitLib",
    "version": "0.0.7",
    "description": "A fast NWG Library",
    "author": "Kotetsu0000",
    'ext_modules': ext_modules,
    'include_dirs': [numpy.get_include()],
    'install_requires' : [
        'numpy',
        'opencv_python',
    ],
    'packages': find_packages(),
}

setup(**setup_kwargs)
