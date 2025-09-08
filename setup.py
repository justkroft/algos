# python setup.py build_ext -i clean
import os
import sys

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
    ("src.tree.treap.randomized_treap", "src/tree/treap/randomized_treap.pyx"),
    ("src.tree.bst.binary_search_tree", "src/tree/bst/binary_search_tree.pyx")
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
            sources=[pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c"
        )
        extensions.append(extension)
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
            compiler_directives=COMPILER_DIRECTIVES,
            language_level=3
        ),
        packages=["src"],
        zip_safe=False
    )


if __name__ == "__main__":
    main()
