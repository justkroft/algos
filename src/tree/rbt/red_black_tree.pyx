cimport cython
from libc.stdlib cimport malloc, realloc, free

import numpy as np
cimport numpy as np

cdef extern from *:
    """
    #ifdef __GNUC__
    #define LIKELY(x) __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define PREFETCH_READ(addr) __builtin_prefetch(addr, 0, 3)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #else
    #define LIKELY(x) (x)
    #define UNLIKELY(x) (x)
    #define PREFETCH_READ(addr)
    #define PREFETCH_WRITE(addr)
    #define FORCE_INLINE inline
    #endif
    """
    bint LIKELY(bint)
    bint UNLIKELY(bint)
    void PREFETCH_READ(void*)
    void PREFETCH_WRITE(void*)

from src.typedefs cimport intp_t
include "src/constants.pxi"
include "src/tree/_base_tree.pxi"

cdef enum Color:
    RED = 0
    BLACK = 1

ctypedef packed struct RBNode_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child
    intp_t parent
    Color color


cdef class RedBlackTree(_BaseTree):

    cdef RBNode_t* nodes

    cdef __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        super().__cinit__(initial_capacity)

        self.nodes = <RBNode_t*>malloc(self.capacity * sizeof(RBNode_t))
        if not self.nodes:
            raise MemoryError("Failed to allocate memory for Red-Black Tree")

        # init all nodes
        cdef intp_t i
        for i in range(self.capacity):
            self.nodes[i].parent = NONE_SENTINEL
            self.nodes[i].color = BLACK

    def __setitem__(self, intp_t key, intp_t value) -> None:
        self.insert(key, value)

    def __delitem__(self, intp_t key) -> None:
        if not self.delete(key):
            raise KeyError(f"Key {key} not found")

    def build_tree(self, keys: list | np.ndarray, values: list | np.ndarray) -> None:
        if len(keys) != len(values):
            raise ValueError("Keys and values must have same length")

        needed_capacity = self._size + len(keys)
        while self.capacity < needed_capacity:
            self._resize_rb_arrays()

        cdef intp_t[:] key_view = np.asarray(keys, dtype=np.int64)
        cdef intp_t[:] val_view = np.asarray(values, dtype=np.int64)
        cdef intp_t n = len(keys)
        cdef intp_t i

        for i in range(n):
            self._insert_node(key_view[i], val_view[i])

    cpdef intp_t delete_multiple(self, key: list | np.ndarray):
        cdef intp_t deleted_count = 0
        cdef intp_t i, n = len(keys)

        for i in range(n):
            if self._delete_node(keys[i]):
                deleted_count += 1
        return deleted_count
