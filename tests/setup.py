import os
from os.path import basename, join

import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import eigency

include_dirs = [np.get_include()]
if "EIGEN_INC" in os.environ:
    useSystemEigen = True
    include_dirs.append(os.environ["EIGEN_INC"])
else:
    useSystemEigen = False

    __eigen_dir__ = eigency.get_eigency_eigen_dir()
    __eigen_lib_dir__ = join(basename(__eigen_dir__), "Eigen")
    include_dirs.append(__eigen_dir__)

extensions = [
    Extension(
        "eigency_tests.eigency_tests",
        sources=["eigency_tests/eigency_tests.pyx"],
        include_dirs=[".", "eigency_tests"] + include_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    extensions.append(
        Extension(
            "eigency_tests.eigency_np2",
            sources=["eigency_tests/eigency_np2.pyx"],
            include_dirs=[".", "eigency_tests"] + include_dirs,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    )

setup(
    name="eigency-tests",
    version="0.0.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives=dict(
            language_level="3",
        ),
    ),
    packages=["eigency_tests"],
)
