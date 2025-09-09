from libc.math cimport INFINITY
from cython cimport boundscheck, wraparound

from src.typedefs cimport intp_t, float64_t

DEF INT_NONE_SENTINEL = -999


cdef class DWayHeap:
    """
    Optimized Cython implementation of a d-way heap data structure.

    A d-way heap is a generalization of a binary heap where each internal node
    has at most d children instead of 2. This implementation provides efficient
    priority queue operations with configurable branching factor.

    Parameters
    ----------
    elements : list, optional
        Initial elements to insert into the heap, by default None
    priorities : list, optional
        Initial priorities corresponding to elements, by default None
    branching_factor : int, optional
        Maximum number of children per internal node, by default 2
    is_max_heap : bool, optional
        Whether to maintain max-heap (True) or min-heap (False) property, 
        by default True

    Attributes
    ----------
    branching_factor : int
        The branching factor of the heap
    is_max_heap : bool
        Whether this is a max heap or min heap

    Raises
    ------
    ValueError
        If elements and priorities lists have different lengths
        If branching_factor is less than 2

    Examples
    --------
    >>> heap = DWayHeap.max_heap([1, 2, 3], [10, 5, 15], branching_factor=3)
    >>> heap.top()  # Returns element with highest priority (3)
    3
    """
    cdef public list _pairs
    cdef public intp_t branching_factor
    cdef public bint is_max_heap
    cdef intp_t _size
    
    def __init__(
        self,
        list elements = None,
        list priorities = None,
        const intp_t branching_factor = 2,
        bint is_max_heap = True
    ):
        if elements is None:
            elements = []
        if priorities is None:
            priorities = []
            
        if len(elements) != len(priorities):
            raise ValueError(
                f"The length of the elements ({len(elements)}) must match the"
                f" length of the priorities ({len(priorities)})."
            )

        if branching_factor < 2:
            raise ValueError(
                f"The branching factor must be greater than 1 ({branching_factor})"
            )

        self.branching_factor = branching_factor
        self.is_max_heap = is_max_heap
        self._size = 0
        self._pairs = []

        if elements:
            self._heapify(elements, priorities)

    def __len__(self) -> intp_t:
        """
        Return the number of elements in the heap.
        
        Returns
        -------
        int
            The current size of the heap
        """
        return self._size

    cpdef bint is_empty(self):
        """
        Check whether the heap is empty.
        
        Returns
        -------
        bool
            True if heap contains no elements, False otherwise
        """
        return self._size == 0

    cpdef object top(self):
        """
        Remove and return the top element from the heap.

        For max heaps, returns the element with highest priority.
        For min heaps, returns the element with lowest priority.
        The heap size is reduced by 1.

        Returns
        -------
        object
            The element with highest (max heap) or lowest (min heap) priority

        Raises
        ------
        RuntimeError
            If the heap is empty
        """
        if self._size == 0:
            raise RuntimeError("The heap is empty!")
        
        if self._size == 1:
            # Single element case - simple removal
            self._size = 0
            return self._pairs.pop()[1]
        
        # Multi-element case
        cdef object result = self._pairs[0][1]
        
        # Move last element to root position
        self._size -= 1  # Decrement size first
        self._pairs[0] = self._pairs[self._size]
        self._pairs.pop()  # Remove the now-duplicate last element
        
        # Restore heap property
        self._sift_down(0)
        
        return result

    cpdef object peek(self):
        """
        Return the top element without removing it from the heap.

        For max heaps, returns the element with highest priority.
        For min heaps, returns the element with lowest priority.
        The heap remains unchanged.

        Returns
        -------
        object
            The element with highest (max heap) or lowest (min heap) priority

        Raises
        ------
        RuntimeError
            If the heap is empty
        """
        if self._size == 0:
            raise RuntimeError("The heap is empty!")
        return self._pairs[0][1]

    cpdef void insert(self, object element, float64_t priority):
        """
        Insert a new element with associated priority into the heap.

        The element is placed at the end of the heap and then bubbled up
        to maintain the heap property.

        Parameters
        ----------
        element : object
            The element to insert
        priority : float64_t
            The priority value associated with the element
        """
        # Add new element at the end
        self._pairs.append((priority, element))
        self._size += 1
        
        # Restore heap property by bubbling up
        self._sift_up(self._size - 1)

    cpdef intp_t first_leaf_index(self):
        """
        Calculate the index of the first leaf node in the heap.
        
        Returns
        -------
        int
            Index of the first leaf node
        """
        if self._size <= 1:
            return 0
        return (self._size - 2) // self.branching_factor + 1

    cdef inline intp_t _parent_index(self, intp_t child_index) nogil:
        """
        Calculate the parent index for a given child index.

        Parameters
        ----------
        child_index : intp_t
            Index of the child node

        Returns
        -------
        intp_t
            Index of the parent node
        """
        return (child_index - 1) // self.branching_factor

    cdef inline intp_t _first_child_index(self, intp_t parent_index) nogil:
        """
        Calculate the index of the first child for a given parent.

        Parameters
        ----------
        parent_index : intp_t
            Index of the parent node

        Returns
        -------
        intp_t
            Index of the first child node
        """
        return parent_index * self.branching_factor + 1

    cdef inline bint _has_priority(self, float64_t a, float64_t b) nogil:
        """
        Compare two priorities according to heap type.

        Parameters
        ----------
        a : float64_t
            First priority value
        b : float64_t
            Second priority value

        Returns
        -------
        bint
            True if 'a' has higher priority than 'b' according to heap type
        """
        if self.is_max_heap:
            return a > b
        else:
            return a < b

    cdef intp_t _find_extreme_child(self, intp_t parent_index):
        """
        Find the child with the most extreme priority (highest for max heap,
        lowest for min heap) among all children of the given parent.

        Parameters
        ----------
        parent_index : intp_t
            Index of the parent node

        Returns
        -------
        intp_t
            Index of child with extreme priority, or INT_NONE_SENTINEL if no children
        """
        cdef intp_t first_child = self._first_child_index(parent_index)
        
        # Check if parent has any children
        if first_child >= self._size:
            return INT_NONE_SENTINEL
            
        cdef intp_t last_child = min(first_child + self.branching_factor, self._size)
        cdef intp_t extreme_idx = first_child
        cdef float64_t extreme_priority = self._pairs[first_child][0]
        cdef intp_t i
        cdef float64_t current_priority
        
        # Find child with most extreme priority
        for i in range(first_child + 1, last_child):
            current_priority = self._pairs[i][0]
            if self._has_priority(current_priority, extreme_priority):
                extreme_priority = current_priority
                extreme_idx = i
                
        return extreme_idx

    cdef void _sift_up(self, intp_t start_index):
        """
        Restore heap property by moving element at start_index upward.

        This is used after insertion to maintain the heap invariant.

        Parameters
        ----------
        start_index : intp_t
            Index of element to sift up
        """
        if start_index == 0:
            return
            
        cdef tuple element = self._pairs[start_index]
        cdef float64_t element_priority = element[0]
        cdef intp_t current_index = start_index
        cdef intp_t parent_idx
        cdef float64_t parent_priority
        
        while current_index > 0:
            parent_idx = self._parent_index(current_index)
            parent_priority = self._pairs[parent_idx][0]
            
            # Stop if heap property is satisfied
            if not self._has_priority(element_priority, parent_priority):
                break
                
            # Move parent down
            self._pairs[current_index] = self._pairs[parent_idx]
            current_index = parent_idx
            
        # Place element in final position
        self._pairs[current_index] = element

    cdef void _sift_down(self, intp_t start_index):
        """
        Restore heap property by moving element at start_index downward.

        This is used after removal to maintain the heap invariant.

        Parameters
        ----------
        start_index : intp_t
            Index of element to sift down
        """
        cdef tuple element = self._pairs[start_index]
        cdef float64_t element_priority = element[0]
        cdef intp_t current_index = start_index
        cdef intp_t extreme_child_idx
        cdef float64_t extreme_child_priority
        cdef intp_t first_leaf = self.first_leaf_index()
        
        # Continue until we reach a leaf node
        while current_index < first_leaf:
            extreme_child_idx = self._find_extreme_child(current_index)
            
            if extreme_child_idx == INT_NONE_SENTINEL:
                break
                
            extreme_child_priority = self._pairs[extreme_child_idx][0]
            
            # Stop if heap property is satisfied
            if not self._has_priority(extreme_child_priority, element_priority):
                break
                
            # Move extreme child up
            self._pairs[current_index] = self._pairs[extreme_child_idx]
            current_index = extreme_child_idx
            
        # Place element in final position
        self._pairs[current_index] = element

    cdef void _heapify(self, list elements, list priorities):
        """
        Build heap from unsorted elements and priorities using Floyd's algorithm.

        This is more efficient than inserting elements one by one.

        Parameters
        ----------
        elements : list
            List of elements to insert
        priorities : list
            List of corresponding priorities
        """
        # Create pairs and set size
        self._pairs = [(priorities[i], elements[i]) for i in range(len(elements))]
        self._size = len(self._pairs)
        
        # No work needed for trivial cases
        if self._size <= 1:
            return
            
        # Start from last internal node and sift down
        cdef intp_t last_internal = self._parent_index(self._size - 1)
        cdef intp_t i
        
        for i in range(last_internal, -1, -1):
            self._sift_down(i)

    cpdef bint _validate(self):
        """
        Validate that the heap property is maintained.

        This is primarily for debugging and testing purposes.

        Returns
        -------
        bool
            True if heap property is satisfied, False otherwise
        """
        if self._size == 0:
            return True
            
        cdef intp_t parent_idx
        cdef intp_t first_child, last_child
        cdef intp_t child_idx
        cdef float64_t parent_priority, child_priority
        cdef intp_t first_leaf = self.first_leaf_index()
        
        # Check each internal node
        for parent_idx in range(first_leaf):
            parent_priority = self._pairs[parent_idx][0]
            first_child = self._first_child_index(parent_idx)
            last_child = min(first_child + self.branching_factor, self._size)
            
            # Check all children of this parent
            for child_idx in range(first_child, last_child):
                child_priority = self._pairs[child_idx][0]
                if self._has_priority(child_priority, parent_priority):
                    return False
                    
        return True

    @classmethod
    def max_heap(cls, elements=None, priorities=None, branching_factor=2):
        """
        Create a max heap instance.

        In a max heap, elements with higher priorities are at the top.

        Parameters
        ----------
        elements : list, optional
            Initial elements, by default None
        priorities : list, optional
            Initial priorities, by default None
        branching_factor : int, optional
            Branching factor, by default 2

        Returns
        -------
        DWayHeap
            A max heap instance
        """
        return cls(elements, priorities, branching_factor, True)
    
    @classmethod  
    def min_heap(cls, elements=None, priorities=None, branching_factor=2):
        """
        Create a min heap instance.

        In a min heap, elements with lower priorities are at the top.

        Parameters
        ----------
        elements : list, optional
            Initial elements, by default None
        priorities : list, optional
            Initial priorities, by default None
        branching_factor : int, optional
            Branching factor, by default 2

        Returns
        -------
        DWayHeap
            A min heap instance
        """
        return cls(elements, priorities, branching_factor, False)