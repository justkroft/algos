cimport cython
import numpy as np
cimport numpy as np


ctypedef Py_ssize_t intp_t


cdef extern from "stdlib.h":
    double drand48()


cdef class TreapNode_t:
    cdef object key
    cdef object value
    cdef double priority
    cdef int size
    cdef TreapNode_t left
    cdef TreapNode_t right
    cdef object __weakref__

    def __cinit__(self, object key, object value):
        self.key = key
        self.value = value
        self.priority = drand48()
        self.size = 1
        self.left = None
        self.right = None


# build the corresponding numpy dtype for TreapNode
# cdef TreapNode_t dummy
# TreapNode = np.asarray(<TreapNode_t[:1]>(&dummy)).dtype


cdef class RandomizedTreap:
    cdef intp_t _size
    cdef TreapNode_t root

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self) -> intp_t:
        return self._size

    def __contains__(self, key) -> bool:
        return self.search(key) is not None

    cpdef bint is_empty(self):
        return self.root is None

    cdef void _update_size(self, TreapNode_t node):
        if node:
            left_size = node.left.size if node.left else 0
            right_size = node.right.size if node.right else 0
            node.size = 1 + left_size + right_size
    
    cdef TreapNode_t _rotate_right(self, TreapNode_t node):
        left_child = node.left
        node.left = left_child.right
        left_child.right = node

        # update size
        self._update_size(node)
        self._update_size(left_child)
        return left_child
    
    cdef TreapNode_t _rotate_left(self, TreapNode_t node):
        right_child = node.right
        node.right = right_child.left
        right_child.left = node

        # update size
        self._update_size(node)
        self._update_size(right_child)
        return right_child
