from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

DEF INT_NONE_SENTINEL = -999
ctypedef Py_ssize_t intp_t


cdef class DWayHeap:
    cdef public list _pairs
    cdef public intp_t branching_factor
    cdef public bint is_max_heap  # True for max heap, False for min heap
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

    def __sizeof__(self) -> intp_t:
        return self._size

    def __len__(self) -> intp_t:
        return self._size

    cpdef bint is_empty(self):
        return self._size == 0

    cpdef object top(self):
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
        if self.is_empty():
            raise RuntimeError("The heap is empty!")
        return self._pairs[0][1]

    cpdef void insert(self, object element, double priority):
        self._pairs.append((priority, element))
        self._size += 1
        self._bubble_up(self._size - 1)

    cpdef intp_t first_leaf_index(self):
        cdef intp_t result

        with nogil:
            result = (self._size - 2) // self.branching_factor + 1
        return result

    cdef intp_t _first_child_index(self, intp_t index) nogil:
        return index * self.branching_factor + 1

    cdef intp_t _parent_index(self, intp_t index) nogil:
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
        """Validate heap property"""
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
