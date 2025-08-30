from libc.math cimport INFINITY
import numpy as np
cimport numpy as np


DEF INT_NONE_SENTINEL = -999
ctypedef Py_ssize_t intp_t


cdef class DWayHeap:
    cdef public list _pairs
    cdef public intp_t branching_factor
    
    def __init__(
        self,
        list elements = [],
        list priorities = [],
        const intp_t branching_factor = 2
    ):
        if len(elements) != len(priorities):
            raise ValueError(
                f"The length of the elements ({len(elements)}) must match the"
                f" length of the priorities ({len(priorities)})."
            )

        if branching_factor < 2:
            raise ValueError(
                f"The branching factor must be greater than 1 ({branching_factor})"
            )

        cdef list _pairs = []  # list of tuples
        self._pairs = _pairs
        self.branching_factor = branching_factor

        if len(elements) > 0:
            self._heapify(elements, priorities)

    def __sizeof__(self) -> intp_t:
        return len(self)

    def __len__(self) -> intp_t:
        return len(self._pairs)

    cpdef bint is_empty(self):
        return len(self) == 0

    cpdef object top(self):
        if self.is_empty():
            raise RuntimeError("...")
        if len(self) == 1:
            return self._pairs.pop()[1]
        else:
            element = self._pairs[0][1]
            self._pairs[0] = self._pairs.pop()
            self._push_down(0)
            return element

    cpdef object peek(self):
        if self.is_empty():
            raise RuntimeError("...")
        return self._pairs[0][1]

    cpdef void insert(self, object element, float priority):
        self._pairs.append((priority, element))
        self._bubble_up(len(self._pairs) - 1)

    cpdef intp_t first_leaf_index(self):
        cdef intp_t size = len(self)
        cdef intp_t result

        with nogil:
            result = (size - 2) // self.branching_factor + 1
        return result

    cdef intp_t _first_child_index(self, intp_t index):
        return index * self.branching_factor + 1

    cdef intp_t _parent_index(self, intp_t index):
        return (index - 1) // self.branching_factor

    cdef intp_t _highest_priority_child_index(
        self, intp_t index
    ) except INT_NONE_SENTINEL:
        cdef intp_t first_index, size, last_index
        cdef intp_t i
        cdef double highest_priority = -INFINITY
        cdef intp_t best_index
        cdef double current_priority
        
        # Calculate indices (can use nogil for arithmetic)
        with nogil:
            first_index = index * self.branching_factor + 1
        
        size = len(self)
        
        with nogil:
            last_index = first_index + self.branching_factor
            if last_index > size:
                last_index = size

        if first_index >= size:
            return INT_NONE_SENTINEL

        best_index = first_index
        
        for i in range(first_index, last_index):
            current_priority = self._pairs[i][0]
            if current_priority > highest_priority:
                highest_priority = current_priority
                best_index = i
                
        return best_index

    cdef void _push_down(self, intp_t index):
        assert (0 <= index < len(self._pairs))
        cdef tuple input_pair = self._pairs[index]
        cdef double input_priority = input_pair[0]
        cdef intp_t current_index = index
        cdef intp_t first_leaf = self.first_leaf_index()
        cdef intp_t child_index
        cdef double child_priority

        while current_index < first_leaf:
            child_index = self._highest_priority_child_index(current_index)
            assert child_index != INT_NONE_SENTINEL 
            
            child_priority = self._pairs[child_index][0]
            if child_priority > input_priority:
                self._pairs[current_index] = self._pairs[child_index]
                current_index = child_index
            else:
                break
                
        self._pairs[current_index] = input_pair

    cdef void _bubble_up(self, intp_t index):
        assert (0 <= index < len(self._pairs))
        cdef tuple input_pair = self._pairs[index]
        cdef double input_priority = input_pair[0]
        cdef intp_t parent_index
        cdef tuple parent
        
        while index > 0:
            parent_index = self._parent_index(index)
            parent = self._pairs[parent_index]

            if input_priority > parent[0]:
                self._pairs[index] = parent
                index = parent_index
            else:
                break
        self._pairs[index] = input_pair

    cdef void _heapify(self, list elements, list priorities):
        assert (len(elements) == len(priorities))
        self._pairs = list(zip(priorities, elements))
        
        cdef intp_t last_inner_node_index, index
        last_inner_node_index = self.first_leaf_index() - 1
        for index in range(last_inner_node_index, -1, -1):
            self._push_down(index)

    cpdef bint _validate(self):
        cdef intp_t current_index = 0
        cdef intp_t first_leaf = self.first_leaf_index()
        cdef float current_priority
        cdef intp_t first_child, last_child_guard

        while current_index < first_leaf:
            current_priority = self._pairs[current_index][0]
            first_child = self._first_child_index(current_index)
            last_child_guard = min(first_child + self.branching_factor, len(self))
            for child_index in range(first_child, last_child_guard):
                if current_priority < self._pairs[child_index][0]:
                    return 0
            current_index += 1
        return 1
        