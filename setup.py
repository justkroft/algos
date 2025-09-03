import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "src.dway_heap.dway_heap",  # Package.module format
        ["src/dway_heap/dway_heap.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "src.treap.randomized_treap",
        ["src/treap/randomized_treap.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions),
    packages=['src'],  # Tell setuptools about the src package
    zip_safe=False,
)


# import setuptools
# import os
# import sys
# from distutils.command.build import build
# from Cython.Build import cythonize

# # python setup.py build_ext -i clean

# DEFINE_MACRO_NUMPY_C_API = (
#     "NPY_NO_DEPRECATED_API",
#     "NPY_1_9_API_VERSION",
# )
# NUMPY_C_API_ARG = f"-D{DEFINE_MACRO_NUMPY_C_API[0]}={DEFINE_MACRO_NUMPY_C_API[1]}". # noqa: E501


# def get_openmp_flag():
#     if sys.platform == "win32":
#         return "/openmp"
#     elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
#         return ""
#     return "-fopenmp"


# class CommandClean(build):
#     def finalize_options(self) -> None:
#         super().finalize_options()
#         __builtins__.__NUMPY_SETUP__ = False
#         try:
#             import numpy
#         except ImportError as e:
#             raise RuntimeError(
#                 "Install NumPy before building this extension"
#             ) from e


# extension_config = {
#     "src": {
#         "name": "src",
#         "sources": ["dway_heap.pyx"],
#         "language": "c",
#         "include_np": True,
#         "extra_compile_args": [NUMPY_C_API_ARG]
#     }
# }

# compiler_directives = {
#     "language_level": 3,
#     "boundscheck": False,
#     "wraparound": False,
#     "initializedcheck": False,
#     "nonecheck": False,
#     "cdivision": True,
#     "profile": False,
# }


# def configure_extensions():
#     import numpy

#     np_include = numpy.get_include()
#     openmp_flag = get_openmp_flag()

#     cython_ext = []
#     for submodule, extension in extension_config.items():
#         sources = []
#         for source in extension["sources"]:
#             source = os.path.join(submodule, source)
#             sources.append(source)
#             include_dirs = []

#             if extension.get("include_np", False):
#                 include_dirs.append(np_include)

#             extra_compile_args = extension.get("extra_compile_args", [])

#             new_ext = setuptools.Extension(
#                 name=extension["name"],
#                 sources=sources,
#                 language=extension.get("language", None),
#                 include_dirs=include_dirs,
#                 extra_compile_args=extra_compile_args
#             )
#             cython_ext.append(new_ext)
#     return cython_ext


# if __name__ == "__main__":
#     ext_modules = configure_extensions()
#     setuptools.setup(
#         ext_modules=cythonize(
#             ext_modules,
#             compiler_directives=compiler_directives
#         ),
#         cmdclass={"build": CommandClean}
#     )
