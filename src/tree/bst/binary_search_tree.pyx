cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset, memcpy
from libc.math cimport log2

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

from src.typedefs cimport intp_t, float64_t


DEF NONE_SENTINEL = -1
DEF INITIAL_CAPACITY = 64
DEF GROWTH_FACTOR = 2
DEF PREFETCH_DISTANCE = 8


ctypedef packed struct Node_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child


cdef class BinarySearchTree:
    cdef Node_t* nodes
    cdef intp_t* free_stack      # Stack of free indices for O(1) allocation
    cdef intp_t capacity
    cdef intp_t _size
    cdef intp_t root_idx
    cdef intp_t free_stack_top
    cdef intp_t free_count

    def __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        self.capacity = initial_capacity
        self.root_idx = NONE_SENTINEL
        self.free_stack_top = -1
        self.free_count = 0
        self._size = 0

        # allocate memory
        self.nodes = <Node_t*>malloc(self.capacity * sizeof(Node_t))
        self.free_stack = <intp_t*>malloc(self.capacity * sizeof(intp_t))

        if not self.nodes or not self.free_stack:
            raise MemoryError("Failed to allocate memory")
        
        # init free stack
        cdef intp_t i
        for i in range(self.capacity):
            self.free_stack[i] = i
        self.free_stack_top = self.capacity - 1
        self.free_count = self.capacity
    
    def __len__(self) -> intp_t:
        return self._size

    def __dealloc__(self):
        if self.nodes:
            free(self.nodes)
        if self.free_stack:
            free(self.free_stack)

    cpdef bint is_empty(self):
        """Check whether the BST is empty."""
        return self._size == 0

    cdef inline intp_t _allocate_node(self) nogil:
        cdef intp_t idx
        if UNLIKELY(self.free_stack_top < 0):
            return NONE_SENTINEL
        
        idx = self.free_stack[self.free_stack_top]
        self.free_stack_top -= 1
        self.free_count -= 1
        return idx

    cdef inline void _deallocate_node(self, intp_t idx) nogil:
        if LIKELY(self.free_stack_top < self.capacity - 1):
            self.free_stack_top += 1
            self.free_stack[self.free_stack_top] = idx
            self.free_count += 1
    
    cdef void _resize_arrays(self):
        cdef intp_t new_capacity = self.capacity * GROWTH_FACTOR
        cdef Node_t* new_nodes = <Node_t*>realloc(
            self.nodes, sizeof(Node_t) * new_capacity
        )
        cdef int* new_free_stack = <intp_t*>realloc(
            self.free_stack, sizeof(intp_t) * new_capacity
        )
        
        if not new_nodes or not new_free_stack:
            raise MemoryError("Failed to resize")

        self.nodes = new_nodes
        self.free_stack = new_free_stack

        # Add new indices to free stack
        cdef intp_t i
        for i in range(self.capacity, new_capacity):
            self.free_stack_top += 1
            self.free_stack[self.free_stack_top] = i
            self.free_count += 1
        
        self.capacity = new_capacity

    cdef inline intp_t _find_node(self, intp_t key) nogil:
        cdef intp_t current = self.root_idx
        cdef Node_t* node
        cdef intp_t next_idx

        while LIKELY(current != NONE_SENTINEL):
            node = &self.nodes[current]

            if LIKELY(node.left_child != NONE_SENTINEL):
                PREFETCH_READ(&self.nodes[node.left_child])
            if LIKELY(node.right_child != NONE_SENTINEL):
                PREFETCH_READ(&self.nodes[node.right_child])

            if LIKELY(key < node.key):
                current = node.left_child
            elif LIKELY(key > node.key):
                current = node.right_child
            else:
                return current

        return NONE_SENTINEL
