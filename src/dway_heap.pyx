import numpy as np
cimport numpy as np


cdef class DWayHeap:
    
    def __init__(
        self,
        list elements,
        list priorities,
        const intp_t branching_factor = 2
    ):
        if len(elements) != len(priorities):
            raise ValueError("...")

        if branching_factor < 2:
            raise ValueError("...")

        cdef list _pairs = []
        self._pairs = []
        self.branching_factor = branching_factor

        if len(elements) > 0:
            self._heapify(elements, priorities)

    cpdef intp_t __sizeof__(self):
        return len(self)

    cpdef intp_t __len__(self):
        return len(self._pairs)

    cpdef intp_t first_leaf_index(self):
        return (len(self) - 2) // self.branching_factor + 1
        
    
