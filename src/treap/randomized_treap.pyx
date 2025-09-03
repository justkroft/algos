cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h":
    double drand48()

import numpy as np
cimport numpy as np

DEF NONE_SENTINEL = -1
DEF INITIAL_CAPACITY = 64
DEF GROWTH_FACTOR = 2

ctypedef Py_ssize_t intp_t


# Array-based node structure - fixed to avoid Python object pointers
ctypedef struct ArrayNode:
    intp_t left_child      # Index of left child (-1 if None)
    intp_t right_child     # Index of right child (-1 if None) 
    double priority        # Random priority for heap property
    intp_t size            # Subtree size for rank operations


cdef class ArrayRandomizedTreap:
    """
    Specialized version for numeric keys using numpy arrays.
    Can perform many operations without GIL since key comparisons are numeric.
    """
    
    cdef ArrayNode* nodes
    cdef double[::1] keys              # Typed memoryview for fast access
    cdef double[::1] _keys_array
    cdef list values                   # Python list for values (arbitrary objects)
    cdef intp_t capacity
    cdef intp_t node_count
    cdef intp_t root_idx
    cdef intp_t _size
    
    def __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        self.capacity = initial_capacity
        self.nodes = <ArrayNode*>malloc(self.capacity * sizeof(ArrayNode))
        
        if not self.nodes:
            raise MemoryError("Failed to allocate numeric treap arrays")
        
        # Use numpy array for numeric keys - much faster than Python list
        self._keys_array = np.empty(initial_capacity, dtype=np.float64)
        self.keys = self._keys_array
        self.values = []
        
        self.node_count = 0
        self.root_idx = NONE_SENTINEL  
        self._size = 0

    def __len__(self) -> intp_t:
        return self._size
    
    def __dealloc__(self):
        if self.nodes:
            free(self.nodes)
    
    cdef void _resize_arrays(self):
        """Resize arrays for numeric version"""
        cdef intp_t new_capacity = self.capacity * GROWTH_FACTOR
        
        # Resize node array (C structs)
        cdef ArrayNode* new_nodes = <ArrayNode*>malloc(new_capacity * sizeof(ArrayNode))
        if not new_nodes:
            raise MemoryError("Failed to resize numeric treap arrays")
        
        # Copy node data
        cdef intp_t i
        for i in range(self.node_count):
            new_nodes[i] = self.nodes[i]
        
        # Resize numpy array for keys
        old_keys_array = np.asarray(self.keys)
        new_keys_array = np.empty(new_capacity, dtype=np.float64)
        new_keys_array[:self.node_count] = old_keys_array[:self.node_count]
        
        free(self.nodes)
        self.nodes = new_nodes
        self._keys_array = new_keys_array
        self.keys = self._keys_array
        self.capacity = new_capacity
    
    cdef intp_t _create_numeric_node(self, double key, object value):
        """Create node with numeric key"""
        if self.node_count >= self.capacity:
            self._resize_arrays()
            
        cdef intp_t idx = self.node_count
        self.node_count += 1
        
        # Initialize C struct
        self.nodes[idx].left_child = NONE_SENTINEL
        self.nodes[idx].right_child = NONE_SENTINEL  
        self.nodes[idx].priority = drand48()
        self.nodes[idx].size = 1
        
        # Store key and value
        self.keys[idx] = key
        self.values.append(value)
        
        return idx
    
    cpdef void insert_numeric(self, double key, object value):
        """Insert with numeric key"""
        self.root_idx = self._insert_numeric_helper(self.root_idx, key, value)
        self._size += 1
    
    cdef intp_t _insert_numeric_helper(self, intp_t node_idx, double key, object value):
        """Numeric insertion helper"""
        if node_idx == NONE_SENTINEL:
            return self._create_numeric_node(key, value)
        
        cdef intp_t new_child_idx
        cdef double child_priority, node_priority, node_key
        
        # Numeric comparison - can be fast
        node_key = self.keys[node_idx]
        
        if key <= node_key:
            new_child_idx = self._insert_numeric_helper(self.nodes[node_idx].left_child, key, value)
            self.nodes[node_idx].left_child = new_child_idx
            
            child_priority = self.nodes[new_child_idx].priority
            node_priority = self.nodes[node_idx].priority
            
            if child_priority > node_priority:
                with nogil:
                    node_idx = self._rotate_right(node_idx)
        else:
            new_child_idx = self._insert_numeric_helper(self.nodes[node_idx].right_child, key, value)
            self.nodes[node_idx].right_child = new_child_idx
            
            child_priority = self.nodes[new_child_idx].priority  
            node_priority = self.nodes[node_idx].priority
            
            if child_priority > node_priority:
                with nogil:
                    node_idx = self._rotate_left(node_idx)
        
        with nogil:
            self._update_size(node_idx)
        return node_idx
    
    cpdef object search_numeric(self, double key):
        """Search for numeric key"""
        cdef intp_t node_idx = self._search_numeric_helper(self.root_idx, key)
        if node_idx == NONE_SENTINEL:
            return None
        return self.values[node_idx]
    
    cdef intp_t _search_numeric_helper(self, intp_t node_idx, double key):
        """Numeric search - much faster than Python object comparison"""
        cdef double node_key
        
        while node_idx != NONE_SENTINEL:
            node_key = self.keys[node_idx]
            
            if key == node_key:
                return node_idx
            elif key < node_key:
                node_idx = self.nodes[node_idx].left_child
            else:
                node_idx = self.nodes[node_idx].right_child
                
        return NONE_SENTINEL
    
    # Rotation and size update methods (same as ArrayBasedTreap)
    cdef intp_t _rotate_right(self, intp_t node_idx) noexcept nogil:
        cdef intp_t left_idx = self.nodes[node_idx].left_child
        self.nodes[node_idx].left_child = self.nodes[left_idx].right_child
        self.nodes[left_idx].right_child = node_idx
        self._update_size(node_idx)
        self._update_size(left_idx)
        return left_idx
    
    cdef intp_t _rotate_left(self, intp_t node_idx) noexcept nogil:
        cdef intp_t right_idx = self.nodes[node_idx].right_child
        self.nodes[node_idx].right_child = self.nodes[right_idx].left_child
        self.nodes[right_idx].left_child = node_idx
        self._update_size(node_idx)
        self._update_size(right_idx)
        return right_idx
    
    cdef void _update_size(self, intp_t node_idx) noexcept nogil:
        if node_idx == NONE_SENTINEL:
            return
        cdef intp_t left_size = 0, right_size = 0
        if self.nodes[node_idx].left_child != NONE_SENTINEL:
            left_size = self.nodes[self.nodes[node_idx].left_child].size
        if self.nodes[node_idx].right_child != NONE_SENTINEL:
            right_size = self.nodes[self.nodes[node_idx].right_child].size
        self.nodes[node_idx].size = 1 + left_size + right_size
