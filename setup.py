import platform
import sys
from setuptools import setup, Extension, find_packages
from setuptools.config import read_configuration
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

if platform.system() == 'Windows':
    extra_args = ['/openmp', "/O2"]
else:
    extra_args = ['-fopenmp', "-O3"]

def get_ext_module(use_openmp:bool):
    if use_Cython:
        ext_modules = cythonize(
            [
                Extension(
                    "VitLib.VitLib_cython.common",
                    ["VitLib/VitLib_cython/common.pyx"],
                    language="c++",
                    extra_compile_args=extra_args if use_openmp else None,
                    extra_link_args=extra_args if use_openmp else None,
                ),
                Extension(
                    "VitLib.VitLib_cython.membrane",
                    ["VitLib/VitLib_cython/membrane.pyx"],
                    language="c++",
                    extra_compile_args=extra_args if use_openmp else None,
                    extra_link_args=extra_args if use_openmp else None,
                    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                ),
                Extension(
                    "VitLib.VitLib_cython.nucleus",
                    ["VitLib/VitLib_cython/nucleus.pyx"],
                    language="c++",
                    extra_compile_args=extra_args if use_openmp else None,
                    extra_link_args=extra_args if use_openmp else None,
                ),
            ],
            compiler_directives={
                'language_level' : "3",
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False,
                'initializedcheck': False,
                'cdivision': True,
            }
        )
    else:
        ext_modules = [
            Extension(
                "VitLib.VitLib_cython.common",
                ["VitLib/VitLib_cython/common.cpp"],
                language="c++",
            ),
            Extension(
                "VitLib.VitLib_cython.membrane",
                ["VitLib/VitLib_cython/membrane.cpp"],
                language="c++",
                extra_compile_args=extra_args if use_openmp else None,
                extra_link_args=extra_args if use_openmp else None,
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            ),
            Extension(
                "VitLib.VitLib_cython.nucleus",
                ["VitLib/VitLib_cython/nucleus.cpp"],
                language="c++",
            ),
        ]
    return ext_modules

def get_setup_kwargs(use_openmp:bool):
    setup_kwargs = {
        "name": "VitLib",
        "version": "3.3.3",
        "description": "A fast and accurate image processing library for cell image analysis.",
        "author": "Kotetsu0000",
        'ext_modules': get_ext_module(use_openmp),
        'include_dirs': [numpy.get_include()],
        'install_requires' : [
            'numpy',
            'opencv_python',
        ],
        'extras_require': {
            'dev': [
                'pytest',
                'Cython',
            ]
        },
        'packages': find_packages(),
    }
    return setup_kwargs

setup_list = [True, False]
for i, setup_data in enumerate(setup_list):
    setup_kwargs = get_setup_kwargs(setup_data)
    try:
        setup(**setup_kwargs)
        break
    except:
        pass
    if i == len(setup_list) - 1:# if the last setup failed
        del setup_kwargs['ext_modules']
        setup(**setup_kwargs)
