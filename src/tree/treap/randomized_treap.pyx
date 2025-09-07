cimport cython
from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h":
    double drand48()

import numpy as np
cimport numpy as np

from src.typedefs cimport intp_t, float64_t

DEF NONE_SENTINEL = -1
# memory management
# We have an initial capacity of 64,
# if this capacity is exceeded, the capacity is doubled through the growth factor
DEF INITIAL_CAPACITY = 64
DEF GROWTH_FACTOR = 2


ctypedef struct ArrayNode_t:
    # Base struct for nodes in RandomizedTreap object
    intp_t left_child   # index of the left child of the node (-1 if None)
    intp_t right_child  # index of the right child of the node (-1 if None)
    float64_t priority  # randomly assigned priority for heap property
    intp_t size         # subtree size for rank operations


cdef class RandomizedTreap:
    """
    Array-based Cython representation of a Randomized Treap (RT). 
    
    An RT is a combination of a Binary Search Tree (BST) and a heap that uses
    randomization to maintain balance.
    BST property:
        * Left children have keys <= parent
        * right children have keys > parent
    Heap property:
        * Each node's priority > it's children's priority
    Due to the randomization maintaining balance with a high probability, the
    major operations have an expected time of O(log n).

    The RT is represented as an array of nodes. The i-th element in the array
    holds information about node 'i'; the element at node 0 is the root of
    the RT.

    Parameters
    ----------
    nodes : ArrayNode_t*
        A pointer to an array (first element of an array) of node structures
    keys : array of float64_t, shape [initial_capacity]
        The keys in the treap. NumPy array for fast speed.
    values : list
        Python list storing the associated values of the keys.

    Notes
    -----
    * In-order traversal gives keys in sorted order (BST).
    * The tree maintains balance probabilistically through random priorities (heap).
    * The random priorities make treap self-balancing with high probability, giving
        expected O(log n) time complexity for search, insert, and remove.
    """
    
    cdef ArrayNode_t* nodes
    cdef float64_t[::1] keys
    cdef float64_t[::1] _keys_array
    cdef list values
    cdef intp_t capacity
    cdef intp_t node_count
    cdef intp_t root_idx
    cdef intp_t _size
    
    def __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        self.capacity = initial_capacity
        # Declare pointer to first element of an array of node structs
        self.nodes = <ArrayNode_t*>malloc(self.capacity * sizeof(ArrayNode_t))
        
        if not self.nodes:
            raise MemoryError("Failed to allocate arrays")
        
        # Use numpy array for keys
        self._keys_array = np.empty(initial_capacity, dtype=np.float64)
        self.keys = self._keys_array
        self.values = []
        
        self.node_count = 0
        self.root_idx = NONE_SENTINEL  
        self._size = 0

    def __len__(self) -> intp_t:
        """Return the size of the randomized treap."""
        return self._size
    
    def __dealloc__(self):
        if self.nodes:
            free(self.nodes)

    cpdef bint is_empty(self):
        """Check whether the treap is empty."""
        return self._size == 0

    cpdef object top(self):
        """
        Remove and return the value with the smallest key
        
        Raises
        ------
        RuntimeError
            Error if the treap is empty.
        """
        if self.is_empty():
            raise RuntimeError("The treap is empty!")

        # Find the minimum key and remove
        cdef float64_t min_key = self.peek()
        cdef object min_value = self.search(min_key)
        self.remove(min_key)

        return min_value

    cpdef float64_t peek(self):
        """
        Return the smallest key without removing it
        
        Raises
        ------
        RuntimeError
            Error if the treap is empty.
        """
        if self.is_empty():
            raise RuntimeError("The treap is empty!")
        cdef intp_t node_idx = self.root_idx

        # Find leftmost node (minimum key)
        while self.nodes[node_idx].left_child != NONE_SENTINEL:
            node_idx = self.nodes[node_idx].left_child
        
        return self.keys[node_idx]

    cpdef intp_t rank(self, float64_t key):
        """
        Returns the 0-based position of a key in sorted order.

        Parameters
        ----------
        key : float64_t
            The key in the treap to rank.

        Returns
        -------
        intp_t : The rank of the key.
        """
        return self._rank_helper(self.root_idx, key)
    
    cdef intp_t _rank_helper(self, intp_t node_idx, float64_t key):
        if node_idx == NONE_SENTINEL:
            return NONE_SENTINEL
        
        cdef float64_t node_key = self.keys[node_idx]
        cdef intp_t left_size = 0
        cdef intp_t right_rank

        if self.nodes[node_idx].left_child != NONE_SENTINEL:
            left_size = self.nodes[self.nodes[node_idx].left_child].size
        
        if key == node_key:
            return left_size
        elif key < node_key:
            return self._rank_helper(self.nodes[node_idx].left_child, key)
        else:
            right_rank = self._rank_helper(self.nodes[node_idx].right_child, key)
            if right_rank == -1:
                return NONE_SENTINEL
            return left_size + 1 + right_rank

    cpdef void remove(self, float64_t key):
        """
        Removes a node while maintaining both BST and heap properties.

        Parameters
        ----------
        key : float64_t
            The key to remove from the treap.

        Raises
        ------
        KeyError
            Error if the provided key is not found in the treap
        """
        cdef intp_t old_size = self._size
        self.root_idx = self._remove_helper(self.root_idx, key)
        if self._size == old_size:
            raise KeyError(f"Key: {key} not found")

    cdef intp_t _remove_helper(self, intp_t node_idx, float64_t key):
        if node_idx == NONE_SENTINEL:
            return NONE_SENTINEL
        
        cdef float64_t node_key = self.keys[node_idx]

        if key < node_key:
            self.nodes[node_idx].left_child = self._remove_helper(
                self.nodes[node_idx].left_child,
                key
            )
        elif key > node_key:
            self.nodes[node_idx].right_child = self._remove_helper(
                self.nodes[node_idx].right_child,
                key
            )
        else:
            self._size -= 1
            return self._remove_node(node_idx)
        
        with nogil:
            self._update_size(node_idx)
        return node_idx

    cdef intp_t _remove_node(self, intp_t node_idx):
        cdef intp_t left_idx = self.nodes[node_idx].left_child
        cdef intp_t right_idx = self.nodes[node_idx].right_child
        
        # Case 1: No children
        if left_idx == NONE_SENTINEL and right_idx == NONE_SENTINEL:
            return NONE_SENTINEL
        
        # Case 2: Only one child
        if left_idx == NONE_SENTINEL:
            return right_idx
        if right_idx == NONE_SENTINEL:
            return left_idx
        
        # Case 3: Two children - rotate to maintain heap property, then recurse
        cdef float64_t left_priority = self.nodes[left_idx].priority
        cdef float64_t right_priority = self.nodes[right_idx].priority
        
        if left_priority > right_priority:
            with nogil:
                node_idx = self._rotate_right(node_idx)
            self.nodes[node_idx].right_child = self._remove_node(
                self.nodes[node_idx].right_child
            )
        else:
            with nogil:
                node_idx = self._rotate_left(node_idx)
            self.nodes[node_idx].left_child = self._remove_node(
                self.nodes[node_idx].left_child
            )
        
        with nogil:
            self._update_size(node_idx)
        return node_idx
    
    cdef intp_t _create_node(self, float64_t key, object value):
        """Create node with key-value pair"""
        if self.node_count >= self.capacity:
            self._resize_arrays()
            
        cdef intp_t idx = self.node_count
        self.node_count += 1
        
        # Initialize C struct
        self.nodes[idx].left_child = NONE_SENTINEL
        self.nodes[idx].right_child = NONE_SENTINEL  
        self.nodes[idx].priority = drand48()
        self.nodes[idx].size = 1
        
        self.keys[idx] = key
        self.values.append(value)
        
        return idx

    cpdef object select(self, intp_t rank):
        """Return the value at the given rank (0-based index)"""
        if rank < 0 or rank >= self._size:
            raise IndexError(f"Rank {rank} out of bounds for treap of size {self._size}")
        
        cdef intp_t node_idx = self._select_helper(self.root_idx, rank)
        return self.values[node_idx]
    
    cdef intp_t _select_helper(self, intp_t node_idx, intp_t target_rank):
        """Find the node at the given rank"""
        cdef intp_t left_size = 0
        if self.nodes[node_idx].left_child != NONE_SENTINEL:
            left_size = self.nodes[self.nodes[node_idx].left_child].size
        
        if target_rank == left_size:
            return node_idx
        elif target_rank < left_size:
            return self._select_helper(self.nodes[node_idx].left_child, target_rank)
        else:
            return self._select_helper(
                self.nodes[node_idx].right_child, 
                target_rank - left_size - 1
            )
    
    cpdef void insert(self, float64_t key, object value):
        """
        Insert a new key into the treao, and generate random priority.
        Adds key-value pairs with rotations to maintain heap property.

        Parameters
        ----------
        key : float64_t
            The new key to add.
        value : object
            The associated value of the new key.
        """
        self.root_idx = self._insert_helper(self.root_idx, key, value)
        self._size += 1
    
    cdef intp_t _insert_helper(self, intp_t node_idx, float64_t key, object value):
        if node_idx == NONE_SENTINEL:
            return self._create_node(key, value)
        
        cdef intp_t new_child_idx
        cdef float64_t child_priority, node_priority, node_key
        
        node_key = self.keys[node_idx]
        
        if key <= node_key:
            new_child_idx = self._insert_helper(self.nodes[node_idx].left_child, key, value)
            self.nodes[node_idx].left_child = new_child_idx
            
            child_priority = self.nodes[new_child_idx].priority
            node_priority = self.nodes[node_idx].priority
            
            if child_priority > node_priority:
                with nogil:
                    node_idx = self._rotate_right(node_idx)
        else:
            new_child_idx = self._insert_helper(self.nodes[node_idx].right_child, key, value)
            self.nodes[node_idx].right_child = new_child_idx
            
            child_priority = self.nodes[new_child_idx].priority  
            node_priority = self.nodes[node_idx].priority
            
            if child_priority > node_priority:
                with nogil:
                    node_idx = self._rotate_left(node_idx)
        
        with nogil:
            self._update_size(node_idx)
        return node_idx
    
    cpdef object search(self, float64_t key):
        """Search for a given key value"""
        cdef intp_t node_idx = self._search_helper(self.root_idx, key)
        if node_idx == NONE_SENTINEL:
            return None
        return self.values[node_idx]
    
    cdef intp_t _search_helper(self, intp_t node_idx, float64_t key):
        cdef float64_t node_key
        
        while node_idx != NONE_SENTINEL:
            node_key = self.keys[node_idx]
            
            if key == node_key:
                return node_idx
            elif key < node_key:
                node_idx = self.nodes[node_idx].left_child
            else:
                node_idx = self.nodes[node_idx].right_child
                
        return NONE_SENTINEL

    cdef intp_t _rotate_right(self, intp_t node_idx) noexcept nogil:
        """Right rotation to maintain heap property after insertion"""
        cdef intp_t left_idx = self.nodes[node_idx].left_child
        self.nodes[node_idx].left_child = self.nodes[left_idx].right_child
        self.nodes[left_idx].right_child = node_idx
        self._update_size(node_idx)
        self._update_size(left_idx)
        return left_idx
    
    cdef intp_t _rotate_left(self, intp_t node_idx) noexcept nogil:
        """Left rotation to maintain heap property after insertion"""
        cdef intp_t right_idx = self.nodes[node_idx].right_child
        self.nodes[node_idx].right_child = self.nodes[right_idx].left_child
        self.nodes[right_idx].left_child = node_idx
        self._update_size(node_idx)
        self._update_size(right_idx)
        return right_idx

    cdef void _resize_arrays(self):
        """Resize arrays"""
        cdef intp_t new_capacity = self.capacity * GROWTH_FACTOR
        
        cdef ArrayNode_t* new_nodes = <ArrayNode_t*>malloc(new_capacity * sizeof(ArrayNode_t))
        if not new_nodes:
            raise MemoryError("Failed to resize arrays")
        
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
    
    cdef void _update_size(self, intp_t node_idx) noexcept nogil:
        """Keeps subtree sizes current for rank operations"""
        if node_idx == NONE_SENTINEL:
            return
        cdef intp_t left_size = 0, right_size = 0
        if self.nodes[node_idx].left_child != NONE_SENTINEL:
            left_size = self.nodes[self.nodes[node_idx].left_child].size
        if self.nodes[node_idx].right_child != NONE_SENTINEL:
            right_size = self.nodes[self.nodes[node_idx].right_child].size
        self.nodes[node_idx].size = 1 + left_size + right_size
