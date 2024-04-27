import sys
from setuptools import setup, Extension, find_packages
import numpy
try:
    from Cython.Build import build_ext, cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    Cython.Compiler.Options.docstrings = True
    Cython.Compiler.Options.embed_pos_in_docstring = True
    use_Cython = True
except:
    use_Cython = False

if use_Cython:
    ext_modules = cythonize(
        [
            Extension(
                "VitLib.VitLib_cython.common",
                ["src/common.pyx"],
                language="c++",
            ),
            Extension(
                "VitLib.VitLib_cython.membrane",
                ["src/membrane.pyx"],
                language="c++",
            ),
            Extension(
                "VitLib.VitLib_cython.nucleus",
                ["src/nucleus.pyx"],
                language="c++",
            ),
        ],
        compiler_directives={
            'language_level' : "3",
            'embedsignature': True
        }
    )
else:
    ext_modules = [
        Extension(
            "VitLib.VitLib_cython.common",
            ["src/common.cpp"],
            language="c++",
        ),
        Extension(
            "VitLib.VitLib_cython.membrane",
            ["src/membrane.cpp"],
            language="c++",
        ),
        Extension(
            "VitLib.VitLib_cython.nucleus",
            ["src/nucleus.cpp"],
            language="c++",
        ),
    ]

try:
    setup_kwargs = {
        "name": "VitLib",
        "version": "2.0.31",
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
except:
    del setup_kwargs['ext_modules']
    setup(**setup_kwargs)
