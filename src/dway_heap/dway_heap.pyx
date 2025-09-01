from libc.math cimport INFINITY
from cython cimport boundscheck, wraparound

DEF INT_NONE_SENTINEL = -999
ctypedef Py_ssize_t intp_t


cdef class DWayHeap:
    """
    Cython implementation of a d-way heap.

    Parameters
    ----------
    elements : list
        The elements of your heap.
    priorities : list
        The priorities associated with your elements.
    branching_factor : int, optional
        The branching factor, the maximum number of children that each internal
        node can have, by default 2. The branching factor must be greater than
        1. A larger branching factor implies a higher number of nodes to be
        checked when pushing a pair on the heap.
    is_max_heap : bool
        Flag to indicate whether the heap is a Max heap (i.e., the elements
        with a higher priority are at the top of the heap) or a Min heap,
        by default True.

    Raises
    ------
    ValueError
        Error if the lengths of the elements and priorities list are not equal.
        Error if the branching factor is less than 2.
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

        self._pairs = []
        self.branching_factor = branching_factor
        self.is_max_heap = is_max_heap
        self._size = 0

        if len(elements) > 0:
            self._heapify(elements, priorities)

    def __len__(self) -> intp_t:
        """Return the size of the heap."""
        return self._size

    cpdef bint is_empty(self):
        """Check whether the heap is empty."""
        return self._size == 0

    cpdef object top(self):
        """
        Return and remove the top element of the heap.

        This method returns the top element and simultaneously removes it
        from the heap, thereby reducing the size by 1.
        
        In case of a max heap, it returns the element with the highest priority
        In case of a min heap, it returns the element with the lowest priority

        Raises
        ------
        RuntimeError
            Error if the heap is empty.
        """
        if self.is_empty():
            raise RuntimeError("The heap is empty!")
        
        cdef tuple result_pair
        cdef object result
        
        if self._size == 1:
            result_pair = self._pairs.pop()
            self._size = 0
            return result_pair[1]
        else:
            result = self._pairs[0][1]
            # Move last element to root and reduce size
            self._pairs[0] = self._pairs[self._size - 1]
            self._pairs.pop()
            self._size -= 1
            self._push_down(0)
            return result

    cpdef object peek(self):
        """
        Return and remove the top element of the heap.

        This method only returns the top element and doesn't alter the heap.
        
        In case of a max heap, it returns the element with the highest priority
        In case of a min heap, it returns the element with the lowest priority

        Raises
        ------
        RuntimeError
            Error if the heap is empty.
        """
        if self.is_empty():
            raise RuntimeError("The heap is empty!")
        return self._pairs[0][1]

    cpdef void insert(self, object element, double priority):
        """
        Add new element-priority pair to the heap.

        Parameters
        ----------
        element : object
            The new element.
        priority : double
            The associated priority of the new element.    
        """
        self._pairs.append((priority, element))
        self._size += 1
        self._bubble_up(self._size - 1)

    cpdef intp_t first_leaf_index(self):
        cdef intp_t result

        with nogil:
            result = (self._size - 2) // self.branching_factor + 1
        return result

    cdef intp_t _first_child_index(self, intp_t index) nogil:
        """
        Compute the first child index of the node.

        Parameters
        ----------
        index : intp_t
            The index of the current node for which we need the child's index.

        Returns
        -------
        intp_t : The index of the left-most child of the heap node.
        """
        return index * self.branching_factor + 1

    cdef intp_t _parent_index(self, intp_t index) nogil:
        """
        Compute the index of the parent of the node.
        
        Parameters
        ----------
        index : intp_t
            The index of the current node for which we need the parent's index.

        Returns
        -------
        intp_t : The index of the parent of the heap node.
        """
        return (index - 1) // self.branching_factor

    cdef inline bint _has_priority(self, double a, double b) nogil:
        """Returns True if 'a' has priority over 'b' based on heap type"""
        if self.is_max_heap:
            return a > b
        else:
            return a < b

    @boundscheck(False)
    @wraparound(False)
    cdef intp_t _highest_priority_child_index(
        self, intp_t index
    ) except INT_NONE_SENTINEL:
        """
        This method fins the child with the highest priority among the children.

        If there is a tie in priority between children, the left-most child is
        returned.

        Parameters
        ----------
        index : intp_t
            The index of the current node for which we need the highest
            priority index of its child.

        Returns
        -------
        intp_t : The index of the highest priority child.
        """
        cdef intp_t first_index, last_index
        cdef intp_t i
        cdef double best_priority, current_priority
        cdef intp_t best_index
        
        with nogil:
            first_index = index * self.branching_factor + 1
            last_index = first_index + self.branching_factor
            if last_index > self._size:
                last_index = self._size

        if first_index >= self._size:
            return INT_NONE_SENTINEL

        best_index = first_index
        best_priority = self._pairs[first_index][0]
        
        # Find child with highest priority
        for i in range(first_index + 1, last_index):
            current_priority = self._pairs[i][0]
            if self._has_priority(current_priority, best_priority):
                best_priority = current_priority
                best_index = i
                
        return best_index

    @boundscheck(False)
    @wraparound(False)
    cdef void _push_down(self, intp_t index):
        """
        This helper method pushes down the heap's root towards its leaf such
        that the modified heap adheres to the invariants. This method is used
        when creating a heap, as well as when removing its 'top' element.

        If an element's child has a higher priority, the element is swapped
        with its child that has the highest priority.

        Parameters
        ----------
        index : intp_t
            The index of the root.
        """
        cdef tuple input_pair = self._pairs[index]
        cdef double input_priority = input_pair[0]
        cdef intp_t current_index = index
        cdef intp_t first_leaf
        cdef intp_t child_index
        cdef double child_priority

        first_leaf = (self._size - 2) // self.branching_factor + 1

        while current_index < first_leaf:
            child_index = self._highest_priority_child_index(current_index)
            if child_index == INT_NONE_SENTINEL:
                break
                
            child_priority = self._pairs[child_index][0]
            if self._has_priority(child_priority, input_priority):
                self._pairs[current_index] = self._pairs[child_index]
                current_index = child_index
            else:
                break
                
        self._pairs[current_index] = input_pair

    @boundscheck(False)
    @wraparound(False)
    cdef void _bubble_up(self, intp_t index):
        """
        This helper method ensures that an inserted element goes to the top if
        necessary to adhere to the heap's invariants.

        If an element has lower priority than it's parent, the current element
        is swapped with its parent.

        Parameters
        ----------
        index : intp_t
            The index of the element to bubble up.
        """
        cdef tuple input_pair = self._pairs[index]
        cdef double input_priority = input_pair[0]
        cdef intp_t parent_index
        cdef tuple parent_pair
        cdef double parent_priority
        
        while index > 0:
            parent_index = (index - 1) // self.branching_factor
                
            parent_pair = self._pairs[parent_index]
            parent_priority = parent_pair[0]

            if self._has_priority(input_priority, parent_priority):
                self._pairs[index] = parent_pair
                index = parent_index
            else:
                break
                
        self._pairs[index] = input_pair

    cdef void _heapify(self, list elements, list priorities):
        """
        Initialize the heap with elements and priorities.
        
        Parameters
        ----------
        elements : list
            The elements of your heap.
        priorities : list
            The priorities associated with your elements.
        """
        self._pairs = list(zip(priorities, elements))
        self._size = len(self._pairs)
        
        # Heapify from last internal node downward
        cdef intp_t last_internal_index
        cdef intp_t i
        
        if self._size <= 1:
            return
            
        with nogil:
            last_internal_index = (self._size - 2) // self.branching_factor
            
        for i in range(last_internal_index, -1, -1):
            self._push_down(i)

    cpdef bint _validate(self):
        """Validate heap properties"""
        cdef intp_t current_index = 0
        cdef intp_t first_leaf
        cdef double current_priority, child_priority
        cdef intp_t first_child, last_child
        cdef intp_t child_index

        with nogil:
            first_leaf = (self._size - 2) // self.branching_factor + 1

        while current_index < first_leaf:
            current_priority = self._pairs[current_index][0]
            
            with nogil:
                first_child = current_index * self.branching_factor + 1
                last_child = first_child + self.branching_factor
                if last_child > self._size:
                    last_child = self._size
                    
            for child_index in range(first_child, last_child):
                child_priority = self._pairs[child_index][0]
                if self._has_priority(child_priority, current_priority):
                    return False
                    
            current_index += 1
        return True

    @classmethod
    def max_heap(cls, elements=None, priorities=None, branching_factor=2):
        """Create a max heap"""
        return cls(elements or [], priorities or [], branching_factor, True)
    
    @classmethod  
    def min_heap(cls, elements=None, priorities=None, branching_factor=2):
        """Create a min heap"""
        return cls(elements or [], priorities or [], branching_factor, False)
