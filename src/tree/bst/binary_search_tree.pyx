"""
Author: Just van der Kroft

This code utilizes some CPU-level compiler optimization hints.

LIKELY/UNLIKELY are used for CPU branch prediction optimization. The CPU
pre-loads instructions for the likely path, reducing pipeline stalls.

PREFETCH_READ/_WRITE are used for memory hierarchy optimization. This brings
data from RAM into L1/L2/L3 cache, and should reduce memory access latency.

See also
https://stackoverflow.com/questions/109710/how-do-the-likely-unlikely-macros-in-the-linux-kernel-work-and-what-is-their-ben. # noqa: E501
http://blog.man7.org/2012/10/how-much-do-builtinexpect-likely-and.html
"""

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

from src.typedefs cimport intp_t, float64_t
include "src/constants.pxi"
include "src/tree/_base_tree.pxi"


cdef class BinarySearchTree(_BaseTree):
    """
    Array-based Cython representation of a Binary Search Tree (BST).

    This implementation can be used as an associative data structure, and is
    highly optimized for (and can only be used with) integer key-value pairs.
    The keys are used for ordering and search, and the values is the data one
    wants to store and retrieve.

    Attributes
    ----------
    nodes : Node_t*
        A pointer to an array (first element of an array) of node structures
        Each node contains a key and a value, and a pointer (index) to its
        left- and right-child.
    """

    def __setitem__(self, intp_t key, intp_t value) -> None:
        """bst[key] = val, same as set()"""
        self.insert(key, value)

    def __delitem__(self, intp_t key) -> None:
        """del bst[key], same as delete()"""
        if not self.delete(key):
            raise KeyError(f"Key {key} not found")

    def build_tree(self, keys: list | np.ndarray, values: list | np.ndarray) -> None:
        """
        Build Tree in optimized manner through an array of keys and values.

        Parameters
        ----------
        keys : list | np.ndarray
            An array of associative keys.
        values : list | np.ndarray
            An array of data you want to store/retrieve.
        """
        if len(keys) != len(values):
            raise ValueError("Keys and values must have same length")

        needed_capacity = self._size + len(keys)
        while self.capacity < needed_capacity:
            self._resize_arrays()
        
        cdef intp_t[:] key_view = np.asarray(keys, dtype=np.int64)
        cdef intp_t[:] val_view = np.asarray(values, dtype=np.int64)
        cdef intp_t n = len(keys)
        cdef intp_t i

        for i in range(n):
            self._insert_node(key_view[i], val_view[i])
    
    cpdef intp_t delete_multiple(self, keys: list | np.ndarray):
        """
        Delete multiple keys and return number of successful deletions.
        
        Parameters
        ----------
        keys : list | np.ndarray
            An array of keys to delete.
        """
        cdef intp_t deleted_count = 0
        cdef intp_t i, n = len(keys)
        
        for i in range(n):
            if self._delete_node(keys[i]):
                deleted_count += 1
        
        return deleted_count

    cpdef bint insert(self, intp_t key, intp_t value):
        """
        Insert key-value pair in the tree.

        Parameters
        ----------
        key : intp_t
            The associative key.
        value : intp_t
            The value associated to the key.

        Returns
        -------
        bint
            Boolean indicating whether the insertion was successfull or not.
        """
        # resize if needed
        if UNLIKELY(self.free_count < 2):
            self._resize_arrays()
        
        cdef intp_t success
        success = self._insert_node(key, value)
        
        return bool(success)
    
    cpdef bint delete(self, intp_t key):
        """
        Delete a key and it's value.

        Parameters
        ----------
        key : intp_t
            The associative key.

        Returns
        -------
        intp_t
            Boolean indicating whether the deletion was successfull or not.
        """
        cdef intp_t success
        success = self._delete_node(key)

        return bool(success)

    cdef intp_t _find_node(self, intp_t key):
        """Find node index for a key (-1 if not found)"""
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

    cdef intp_t _insert_node(self, intp_t key, intp_t value):
        """Internal insertion logic"""
        cdef intp_t new_idx

        if UNLIKELY(self.root_idx == NONE_SENTINEL):
            new_idx = self._allocate_node()
            if UNLIKELY(new_idx == NONE_SENTINEL):
                # require resize
                return 0
            
            self.nodes[new_idx].key = key
            self.nodes[new_idx].value = value
            self.nodes[new_idx].left_child = NONE_SENTINEL
            self.nodes[new_idx].right_child = NONE_SENTINEL
            self.root_idx = new_idx
            self._size += 1
            return 1
        
        cdef intp_t current = self.root_idx
        cdef intp_t parent = NONE_SENTINEL
        cdef Node_t* node
        cdef bint go_left = False

        while LIKELY(current != NONE_SENTINEL):
            parent = current
            node = &self.nodes[current]

            # prefect likely next node
            if LIKELY(key < node.key):
                if LIKELY(node.left_child != NONE_SENTINEL):
                    PREFETCH_READ(&self.nodes[node.left_child])
                current = node.left_child
                go_left = True
            elif LIKELY(key > node.key):
                if LIKELY(node.right_child != NONE_SENTINEL):
                    PREFETCH_READ(&self.nodes[node.right_child])
                current = node.right_child
                go_left = False
            else:
                node.value = value
                return 1
        
        # allocate and link new node
        new_idx = self._allocate_node()
        if UNLIKELY(new_idx == NONE_SENTINEL):
            # require resize
            return 0
        
        self.nodes[new_idx].key = key
        self.nodes[new_idx].value = value
        self.nodes[new_idx].left_child = NONE_SENTINEL
        self.nodes[new_idx].right_child = NONE_SENTINEL

        # link to parent
        if go_left:
            self.nodes[parent].left_child = new_idx
        else:
            self.nodes[parent].right_child = new_idx
        
        self._size += 1
        return 1

    cdef intp_t _delete_node(self, intp_t key):
        """Internal deletion logic"""
        cdef intp_t current = self.root_idx
        cdef intp_t parent = NONE_SENTINEL
        cdef bint is_left_child = False
        cdef Node_t* node

        cdef intp_t child
        cdef intp_t successor
        cdef intp_t successor_parent

        # find node to delete
        while LIKELY(current != NONE_SENTINEL):
            node = &self.nodes[current]

            if LIKELY(key < node.key):
                parent = current
                current = node.left_child
                is_left_child = True
            elif LIKELY(key > node.key):
                parent = current
                current = node.right_child
                is_left_child = False
            else:
                # node not found
                break
        
        if UNLIKELY(current == NONE_SENTINEL):
            # node not found
            return 0
        
        node = &self.nodes[current]

        # leaf node
        if LIKELY(
            node.left_child == NONE_SENTINEL
            and node.right_child == NONE_SENTINEL
        ):
            if UNLIKELY(parent == NONE_SENTINEL):
                self.root_idx = NONE_SENTINEL
            elif is_left_child:
                self.nodes[parent].left_child = NONE_SENTINEL
            else:
                self.nodes[parent].right_child = NONE_SENTINEL
        
        # one child
        elif LIKELY(
            node.left_child == NONE_SENTINEL
            or node.right_child == NONE_SENTINEL
        ):
            child = node.left_child if node.left_child != NONE_SENTINEL else node.right_child

            if UNLIKELY(parent == NONE_SENTINEL):
                self.root_idx = child
            elif is_left_child:
                self.nodes[parent].left_child = child
            else:
                self.nodes[parent].right_child = child
        
        # two children, find inorder successor
        else:
            successor = node.right_child
            successor_parent = current

            # find left-most node in right subtree
            while LIKELY(self.nodes[successor].left_child != NONE_SENTINEL):
                successor_parent = successor
                successor = self.nodes[successor].left_child
            
            node.key = self.nodes[successor].key
            node.value = self.nodes[successor].value

            # delete sucessor (at most one child)
            if successor_parent == current:
                self.nodes[current].right_child = self.nodes[successor].right_child
            else:
                self.nodes[successor_parent].left_child = self.nodes[successor].right_child

            current = successor  # the node that gets deleted
        
        self._deallocate_node(current)
        self._size -= 1
        return 1
