import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "NumPy is required to build this package. Please install it first."
    )

COMPILER_DIRECTIVES = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "nonecheck": False,
    "cdivision": True,
    "profile": False,
}

NUMPY_C_API = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")
]

pyx_files = [
    ("src.dway_heap.dway_heap", "src/dway_heap/dway_heap.pyx"),
    ("src.treap.randomized_treap", "src/treap/randomized_treap.pyx")
]


def get_openmp_flags() -> tuple[list]:
    """Get OpenMP compiler and linker flags based on platform."""
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        # macOS with Homebrew libomp
        if "openmp" in os.getenv("CPPFLAGS", ""):
            return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
        return [], []
    else:  # noqa: RET505
        # Linux and other Unix-like systems
        return ["-fopenmp"], ["-fopenmp"]


def create_extensions(
    pyx_files: list[tuple],
    enable_openmp: bool = True
) -> list[Extension]:
    """
    Create Cython extension for all available .pyx files.

    Parameters
    ----------
    pyx_files : list[tuple]
        A list of tuples. The first element of the tuple is the .pyx file in
        `Package.module` format. The second element is the `path` to the file.
    enable_openmp : bool
        Flag for enabling openmp when compiling, by default True.

    Returns
    -------
    list[Extension]
        A list of Cython extensions
    """
    extensions = []
    openmp_compile_flags, openmp_link_flags = (
        get_openmp_flags()
        if enable_openmp
        else ([], [])
    )

    for module_name, pyx_path in pyx_files:
        extra_compile_args = [
            f"-D{name}={value}"
            for name, value in NUMPY_C_API
        ]
        extra_link_args = []

        if enable_openmp and openmp_compile_flags:
            extra_compile_args.extend(openmp_compile_flags)
            extra_link_args.extend(openmp_link_flags)

        if sys.platform != "win32":
            extra_compile_args.extend(["-O3", "-ffast-math"])

        extension = Extension(
            name=module_name,
            source=[pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c"
        )
        extensions.extend(extension)
    return extensions


def main() -> None:
    """Main setup function for compiling"""
    # Filter out non-existing files
    files = [
        (name, path)
        for name, path in pyx_files
        if os.path.exists(path)
    ]

    if not files:
        raise RuntimeError("No .pyx files found to compile")

    extensions = create_extensions(files)

    setup(
        ext_modules=cythonize(
            extensions,
            comiler_directives=COMPILER_DIRECTIVES,
            packages=["src"],
            zip_safe=False,
        )
    )


if __name__ == "__main__":
    main()


# import numpy as np
# from Cython.Build import cythonize
# from setuptools import Extension, setup

# extensions = [
#     Extension(
#         "src.dway_heap.dway_heap",  # Package.module format
#         ["src/dway_heap/dway_heap.pyx"],
#         include_dirs=[np.get_include()],
#     ),
#     Extension(
#         "src.treap.randomized_treap",
#         ["src/treap/randomized_treap.pyx"],
#         include_dirs=[np.get_include()]
#     )
# ]

# setup(
#     ext_modules=cythonize(extensions),
#     packages=['src'],  # Tell setuptools about the src package
#     zip_safe=False,
# )


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
