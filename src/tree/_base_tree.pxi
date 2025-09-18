ctypedef packed struct Node_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child


cdef class _BaseTree:
    """
    Base class for array-based tree implementations.
    
    Contains common functionality like memory management, traversals, 
    and basic operations that can be shared across different tree variants.

    Tree algorithms that inherit from this base class are expected to implement
    the following 'private' methods by itself:
        - _find_node(self, intp_t key):
            Find node index for a key
        - _insert_node(self, intp_t key, intp_t value):
            Internal tree insertion logic
        - _delete_node(self, intp_t key):
            Internal tree deletion logic
        - _resize_arrays(self):
            Resize the arrays of nodes in line with tree logic
    
    Attributes
    ----------
    nodes : Node_t*
        A pointer to an array of node structures
    free_stack : intp_t*
        Stack of free indices for O(1) allocation
    capacity : intp_t
        Current capacity of the arrays
    _size : intp_t
        Number of elements currently in the tree
    root_idx : intp_t
        Index of the root node
    free_stack_top : intp_t
        Top index of the free stack
    free_count : intp_t
        Number of free nodes available
    """
    cdef Node_t* nodes
    cdef intp_t* free_stack
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
        """len(bst), return number of elements"""
        return self._size

    def __contains__(self, intp_t key) -> intp_t:
        """Key in bst, same as contains()"""
        return self.contains(key)

    def __getitem__(self, intp_t key) -> intp_t:
        """bst[key], same as get()"""
        return self.get(key)

    def __bool__(self) -> bint:
        """bool(bst), True of not empty"""
        return self._size > 0

    def __dealloc__(self):
        if self.nodes:
            free(self.nodes)
        if self.free_stack:
            free(self.free_stack)

    def __setitem__(self, intp_t key, intp_t value) -> None:
        """bst[key] = val, same as set()"""
        self.insert(key, value)

    def __delitem__(self, intp_t key) -> None:
        """del bst[key], same as delete()"""
        if not self.delete(key):
            raise KeyError(f"Key {key} not found")
    
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

    # memory management methods
    cdef inline intp_t _allocate_node(self):
        """Get free node from free stack (O(1))"""
        cdef intp_t idx
        if UNLIKELY(self.free_stack_top < 0):
            return NONE_SENTINEL
        
        idx = self.free_stack[self.free_stack_top]
        self.free_stack_top -= 1
        self.free_count -= 1
        return idx

    cdef inline void _deallocate_node(self, intp_t idx):
        """Return node to free stack (O(1))"""
        if LIKELY(self.free_stack_top < self.capacity - 1):
            self.free_stack_top += 1
            self.free_stack[self.free_stack_top] = idx
            self.free_count += 1

    # public methods
    cpdef bint is_empty(self):
        """Check whether the BST is empty."""
        return self._size == 0

    cpdef intp_t get(self, intp_t key):
        """
        Retrieve value for a key.

        Parameters
        ----------
        key : intp_t
            The associative key.

        Returns
        -------
        intp_t
            The value associated to the key.
        
        Raises
        ------
        KeyError
            Error if the key is not found.
        """
        cdef intp_t idx
        idx = self._find_node(key)
        
        if idx == NONE_SENTINEL:
            raise KeyError(f"Key {key} not found")
        
        return self.nodes[idx].value

    cpdef bint contains(self, intp_t key):
        """
        Check existence of a value for a key.

        Parameters
        ----------
        key : intp_t
            The associative key.

        Returns
        -------
        bint
            Boolean indicating whether the key exists in the tree or not.
        """
        cdef intp_t idx
        idx = self._find_node(key)
        return idx != NONE_SENTINEL

    cpdef np.ndarray get_multiple(self, keys: list | np.ndarray):
        """
        Get multiple values efficiently, returns NumPy array with NONE_SENTINEL
        for missing keys. I.e., if the key does not exist, the array will
        contain -1 as a value.

        - For small inputs (len(keys) < 32), a simple unsorted loop is used.
        This avoids the O(n log n) sort overhead and is faster when the
        number of lookups is small.
        - For larger inputs, keys are sorted before lookup. This improves
        cache locality in the binary search tree traversal (`_find_node()`),
        since lookups of nearby keys tend to reuse nodes higher in the tree.
        Prefetching is also more effective, reducing cache misses.

        Parameters
        ----------
        keys : list | np.ndarray
            An array of keys to retrieve.

        Returns
        -------
        np.ndarray[int64]
            Array of values corresponding to the input keys, with
            missing keys marked as NONE_SENTINEL.
        """
        cdef intp_t n = len(keys)
        if n == 0:
            return np.array([], dtype=np.int64)

        # heuristic cutoff
        cdef intp_t[:] results
        cdef intp_t i, idx
        if n < 32:
            results = np.empty(n, dtype=np.int64)
            for i in range(n):
                idx = self._find_node(keys[i])
                results[i] = self.nodes[idx].value if idx != NONE_SENTINEL else NONE_SENTINEL
            return np.asarray(results)

        # larger input array
        # sort for cache-friendly traversal
        cdef intp_t[:] sorted_indices = np.argsort(keys)
        cdef intp_t[:] sorted_keys = keys[sorted_indices]
        results = np.empty_like(keys)

        cdef intp_t original_pos
        for i in range(n):
            idx = self._find_node(sorted_keys[i])
            original_pos = sorted_indices[i]
            results[original_pos] = self.nodes[idx].value if idx != NONE_SENTINEL else NONE_SENTINEL
        return np.asarray(results)
    
    cpdef np.ndarray contains_multiple(self, keys: list | np.ndarray):
        """
        Check existence of multiple keys efficiently, returns boolean NumPy
        array: 1 if the key exists in the tree, 0 if not.

        Parameters
        ----------
        keys : list | np.ndarray
            An array of keys to check for existence.
        """
        cdef intp_t n = len(keys)
        cdef np.uint8_t[:] result = np.empty(n, dtype=np.uint8)
        cdef intp_t i, idx
        
        for i in range(n):
            idx = self._find_node(keys[i])
            result[i] = 1 if idx != NONE_SENTINEL else 0
        
        return np.asarray(result, dtype=bool)

    # public methods: tree traversals
    cpdef np.ndarray keys(self):
        """Return all keys in sorted order (inorder traversal)."""
        return self.inorder()

    cpdef np.ndarray values(self):
        """Return all values in key-sorted order."""
        if self._size == 0:
            return np.array([], dtype=np.int64)

        cdef intp_t[:] result = np.empty(self._size, dtype=np.int64)
        cdef intp_t result_idx = 0

        self._inorder_values_traversal(self.root_idx, result, &result_idx)
        return np.asarray(result)

    cpdef list items(self):
        """Return all (key, value) pairs in sorted order."""
        return self.inorder_items()

    cpdef np.ndarray inorder(self):
        """Return inorder traversal of keys"""
        if self._size == 0:
            return np.array([], dtype=np.int64)

        cdef intp_t[:] result = np.empty(self._size, dtype=np.int64)
        cdef intp_t result_idx = 0

        self._inorder_traversal(self.root_idx, result, &result_idx)
        return np.asarray(result)

    cpdef np.ndarray preorder(self):
        """Return preorder traversal of keys"""
        if self._size == 0:
            return np.array([], dtype=np.int64)

        cdef intp_t[:] result = np.empty(self._size, dtype=np.int64)
        cdef intp_t result_idx = 0

        self._preorder_traversal(self.root_idx, result, &result_idx)
        return np.asarray(result)

    cpdef np.ndarray postorder(self):
        """Return postorder traversal of keys"""
        if self._size == 0:
            return np.array([], dtype=np.int64)

        cdef intp_t[:] result = np.empty(self._size, dtype=np.int64)
        cdef intp_t result_idx = 0

        self._postorder_traversal(self.root_idx, result, &result_idx)
        return np.asarray(result)

    cpdef list inorder_items(self):
        """Return inorder traversal of (key, value) pairs"""
        if self._size == 0:
            return []
        
        cdef list result = []
        self._inorder_items_traverse(self.root_idx, result)
        return result

    cpdef list preorder_items(self):
        """Return preorder traversal of (key, value) pairs"""
        if self._size == 0:
            return []
        
        cdef list result = []
        self._preorder_items_traverse(self.root_idx, result)
        return result

    cpdef list postorder_items(self):
        """Return postorder traversal of (key, value) pairs"""
        if self._size == 0:
            return []
        
        cdef list result = []
        self._postorder_items_traverse(self.root_idx, result)
        return result

    cdef void _inorder_traversal(
        self,
        intp_t node_idx,
        intp_t[:] result,
        intp_t* result_idx
    ):
        """Left, root, right"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]

        if node.left_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.left_child])

        self._inorder_traversal(node.left_child, result, result_idx)
        result[result_idx[0]] = node.key
        result_idx[0] += 1

        if node.right_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.right_child])

        self._inorder_traversal(node.right_child, result, result_idx)
    
    cdef void _inorder_values_traversal(
        self,
        intp_t node_idx,
        intp_t[:] result,
        intp_t* result_idx
    ):
        """
        Left, root, right.

        Same method as `_inorder_traversal()`, but instead this method traverses
        the values rather than the keys. Helper method for `values()`.
        """
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]

        if node.left_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.left_child])

        self._inorder_values_traversal(node.left_child, result, result_idx)
        result[result_idx[0]] = node.value
        result_idx[0] += 1

        if node.right_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.right_child])

        self._inorder_values_traversal(node.right_child, result, result_idx)

    cdef void _preorder_traversal(
        self,
        intp_t node_idx,
        intp_t[:] result,
        intp_t* result_idx
    ):
        """Root, left, right"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]

        result[result_idx[0]] = node.key
        result_idx[0] += 1

        if node.left_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.left_child])
        if node.right_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.right_child])

        self._preorder_traversal(node.left_child, result, result_idx)
        self._preorder_traversal(node.right_child, result, result_idx)

    cdef void _postorder_traversal(
        self,
        intp_t node_idx,
        intp_t[:] result,
        intp_t* result_idx
    ):
        """Left, right, root"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]

        if node.left_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.left_child])
        if node.right_child != NONE_SENTINEL:
            PREFETCH_READ(&self.nodes[node.right_child])
        
        self._postorder_traversal(node.left_child, result, result_idx)
        self._postorder_traversal(node.right_child, result, result_idx)
        result[result_idx[0]] = node.key
        result_idx[0] += 1

    cdef void _inorder_items_traverse(self, intp_t node_idx, list result):
        """Left, root, right"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]
        
        self._inorder_items_traverse(node.left_child, result)
        result.append((node.key, node.value))
        self._inorder_items_traverse(node.right_child, result)

    cdef void _preorder_items_traverse(self, intp_t node_idx, list result):
        """Root, left, right"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]
        
        result.append((node.key, node.value))
        self._preorder_items_traverse(node.left_child, result)
        self._preorder_items_traverse(node.right_child, result)

    cdef void _postorder_items_traverse(self, intp_t node_idx, list result):
        """Left, right, root"""
        if node_idx == NONE_SENTINEL:
            return
        
        cdef Node_t* node = &self.nodes[node_idx]
        
        self._postorder_items_traverse(node.left_child, result)
        self._postorder_items_traverse(node.right_child, result)
        result.append((node.key, node.value))

    # range queries
    cpdef tuple range_query(self, intp_t min_key, intp_t max_key):
        """
        Range query returning arrays for keys and values.

        Parameters
        ----------
        min_key : intp_t
            The mininum key in the desired range.
        max_key : intp_t
            The maximum key in the desired range.
        
        Returns
        -------
        tuple
            (keys_array, values_array) as arrays.
        """
        if self._size == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty)

        cdef intp_t count = self._count_range(self.root_idx, min_key, max_key)

        if count == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty)

        cdef intp_t[:] keys = np.empty(count, dtype=np.int64)
        cdef intp_t[:] values = np.empty(count, dtype=np.int64)
        cdef intp_t idx = 0

        self._range_query_fill(self.root_idx, min_key, max_key, keys, values, &idx)
        return (np.asarray(keys), np.asarray(values))

    cdef intp_t _count_range(self, intp_t node_idx, intp_t min_key, intp_t max_key):
        """Count nodes in range"""
        if node_idx == NONE_SENTINEL:
            return 0

        cdef Node_t* node = &self.nodes[node_idx]
        cdef intp_t count = 0

        if node.key > max_key:
            return self._count_range(node.left_child, min_key, max_key)
        elif node.key < min_key:
            return self._count_range(node.right_child, min_key, max_key)
        else:
            count = 1
            count += self._count_range(node.left_child, min_key, max_key)
            count += self._count_range(node.right_child, min_key, max_key)
            return count

    cdef void _range_query_fill(
        self,
        intp_t node_idx,
        intp_t min_key,
        intp_t max_key,
        intp_t[:] keys,
        intp_t[:] values,
        intp_t* idx
    ):
        """
        Fill pre-allocated arrays.

        Get the number of nodes in the range using `_count_range()`.
        Pass empty numpy arrays with this size and fill.
        """
        if node_idx == NONE_SENTINEL:
            return

        cdef Node_t* node = &self.nodes[node_idx]

        if node.key > max_key:
            self._range_query_fill(node.left_child, min_key, max_key, keys, values, idx)
        elif node.key < min_key:
            self._range_query_fill(node.right_child, min_key, max_key, keys, values, idx)
        else:
            self._range_query_fill(node.left_child, min_key, max_key, keys, values, idx)

            keys[idx[0]] = node.key
            values[idx[0]] = node.value
            idx[0] += 1
            
            self._range_query_fill(node.right_child, min_key, max_key, keys, values, idx)

    # basic tree operations
    # to be overridden by subclass
    cpdef void build_tree(self, keys: list | np.ndarray, values: list | np.ndarray):
        raise NotImplementedError("Subclass must implement build_tree")

    cdef intp_t _find_node(self, intp_t key):
        raise NotImplementedError("Subclasses must implement _find_node")

    cdef intp_t _insert_node(self, intp_t key, intp_t value):
        raise NotImplementedError("Subclasses must implement _insert_node")

    cdef intp_t _delete_node(self, intp_t key):
        raise NotImplementedError("Subclasses must implement _delete_node")

    cdef void _resize_arrays(self):
        raise NotImplementedError("Subclasses must implement _resize_arrays")
