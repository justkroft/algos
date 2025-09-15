ctypedef packed struct Node_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child


cdef class _BaseTree:
    pass
