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

cdef enum NodeColor:
    RED = 0
    BLACK = 1

ctypedef packed struct RBNode_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child
    intp_t parent
    NodeColor color


cdef class RedBlackTree(_BaseTree):

    cdef RBNode_t* rb_nodes

    cdef __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        super(RedBlackTree, self).__cinit__(initial_capacity)

        # free base nodes from _BaseTree and allocate RBNodes
        if self.nodes:
            free(self.nodes)

        self.rb_nodes = <RBNode_t*>malloc(self.capacity * sizeof(RBNode_t))
        if not self.rb_nodes:
            raise MemoryError("Failed to allocate memory for Red-Black Tree")

        # for compatibility with base class
        self.nodes = <Node_t*>self.rb_nodes

    def __dealloc__(self):
        if self.rb_nodes:
            free(self.rb_nodes)
        # Set to NULL to prevent double free in base class
        self.nodes = NULL
        if self.free_stack:
            free(self.free_stack)
        self.free_stack = NULL

    cdef intp_t _find_node(self, intp_t key):
        cdef intp_t current = self.root_idx
        cdef RBNode_t* node
    
        while LIKELY(current != NONE_SENTINEL):
            node = &self.rb_nodes[current]

            if LIKELY(node.left_child != NONE_SENTINEL):
                PREFETCH_READ(&self.rb_nodes[node.left_child])
            if LIKELY(node.right_child != NONE_SENTINEL):
                PREFETCH_READ(&self.rb_nodes[node.right_child])

            if LIKELY(key < node.key):
                current = node.left_child
            elif LIKELY(key > node.key):
                current = node.right_child
            else:
                return current

        return NONE_SENTINEL

    cdef intp_t _insert_node(self, intp_t key, intp_t value):
        cdef intp_t new_idx = self._allocate_node()
        if UNLIKELY(new_idx == NONE_SENTINEL)
            return 0

        # init new rb-node
        self.rb_nodes[new_idx].key = key
        self.rb_nodes[new_idx].value = value
        self.rb_nodes[new_idx].left_child = NONE_SENTINEL
        self.rb_nodes[new_idx].right_child = NONE_SENTINEL
        self.rb_nodes[new_idx].parent = NONE_SENTINEL
        self.rb_nodes[new_idx].color = RED

        if UNLIKELY(self.root_idx == NONE_SENTINEL):  # empty tree
            self.root_idx = new_idx
            self.rb_nodes[new_idx].color = BLACK  # root is always black
            self._size += 1
            return 1

        cdef intp_t current = self.root_idx
        cdef intp_t parent = NONE_SENTINEL
        cdef RBNode_t* node

        while LIKELY(current != NONE_SENTINEL):
            parent = current
            node = &self.rb_nodes[current]

            if LIKELY(key < node.key):
                if LIKELY(node.left_child != NONE_SENTINEL):
                    PREFETCH_READ(&self.rb_nodes[node.left_child])
                current = node.left_child
            elif LIKELY(key > node.key):
                if LIKELY(node.right_child != NONE_SENTINEL):
                    PREFETCH_READ(&self.rb_nodes[node.right_child])
                current = node.right_child
            else:
                # key exists and we update value
                node.value = value
                self._deallocate_node(new_idx)
                return 1

        self.rb_nodes[new_idx].parent = parent
        if key < self.rb_nodes[parent].key:
            self.rb_nodes[parent].left_child = new_idx
        else:
            self.rb_nodes[parent].right_child = new_idx
    
        self._insert_fix(new_idx)
        self._size += 1
        return 1

    cdef void _insert_fix(self, intp_t node_idx):
        cdef intp_t current = node_idx
        cdef intp_t parent, grandparent, uncle

        while (
            current != self.root_idx
            and self.rb_nodes[self.rb_nodes[current].parent].color == RED
        ):
            parent = self.rb_nodes[current].parent
            grandparent = self.rb_nodes[parent].parent

            if parent == self.rb_nodes[grandparent].left_child:
                uncle = self.rb_nodes[grandparent].right_child

                if (
                    uncle != NONE_SENTINEL
                    and self.rb_nodes[uncle] == RED
                ):  # uncle is red
                    self.rb_nodes[parent].color = BLACK
                    self.rb_nodes[uncle].color = BLACK
                    self.rb_nodes[grandparent].color = RED
                    current = grandparent
                else:  # uncle is black, current is right child
                    if current == self.rb_nodes[parent].right_child:
                        current = parent
                        self._rotate_left(current)
                        parent = self.rb_nodes[current].parent
                        grandparent = self.rb_nodes[parent].parent

                    # uncle is black, current is left child
                    self.rb_nodes[parent].color = BLACK
                    self.rb_nodes[grandparent].color = RED
                    self._rotate_right(grandparent)
            else:
                uncle = self.rb_nodes[grandparent].left_child

                if (
                    uncle != NONE_SENTINEL
                    and self.rb_nodes[uncle].color == RED
                ):  # uncle is red
                    self.rb_nodes[parent].color = BLACK
                    self.rb_nodes[uncle].color = BLACK
                    self.rb_nodes[grandparent].color = RED
                    current = grandparent
                else:
                    # uncle is black, current is left child
                    if current == self.rb_nodes[parent].left_child:
                        current = parent
                        self._rotate_right(current)
                        parent = self.rb_nodes[current].parent
                        grandparent = self.rb_nodes[parent].parent

                    # uncle is balck, current is right child
                    self.rb_nodes[parent].color = BLACK
                    self.rb_nodes[grandparent].color = RED
                    self._rotate_left(grandparent)

        # root is always black
        self.rb_nodes[self.root_idx].color = BLACK
