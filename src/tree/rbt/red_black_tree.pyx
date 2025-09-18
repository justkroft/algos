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

from src.typedefs cimport intp_t
include "src/constants.pxi"
include "src/tree/_base_tree.pxi"

cdef intp_t NIL_SENTINEL = -2

cdef enum NodeColor:
    RED = 0b0
    BLACK = 0b1

cdef enum NodeFlags:
    COLOR_MASK = 0b1
    DELETED_FLAG = 0b10
    MODIFIED_FLAG = 0b100

ctypedef packed struct RBNode_t:
    intp_t key
    intp_t value
    intp_t left_child
    intp_t right_child
    intp_t parent
    unsigned char flags


cdef class RedBlackTree(_BaseTree):

    cdef RBNode_t* rb_nodes
    cdef intp_t nil_node_idx

    cdef __cinit__(self, intp_t initial_capacity=INITIAL_CAPACITY):
        super(RedBlackTree, self).__cinit__(initial_capacity)

        # free base nodes from _BaseTree and allocate RBNodes
        if self.nodes:
            free(self.nodes)

        self.rb_nodes = <RBNode_t*>malloc(self.capacity * sizeof(RBNode_t))
        if not self.rb_nodes:
            raise MemoryError("Failed to allocate memory for Red-Black Tree")

        # for compatibility with base class
        self.nodes = <Node_t*>self.rb_nodes

        # NIL_SENTINEL at index 0
        self.nil_node_idx = 0
        self.rb_nodes[0].flags = BLACK  # NIL is always a black node
        self.rb_nodes[0].parent = 0
        self.rb_nodes[0].left_child = 0
        self.rb_nodes[0].right_child = 0

        # start free stack from index 1
        self.free_stack_top = self.capacity - 2  # 0 based and NIL node
        self.free_count = self.capacity - 1
        cdef intp_t i
        for i in range(1, self.capacity):
            self.free_stack[i-1] = i

    def __dealloc__(self):
        if self.rb_nodes:
            free(self.rb_nodes)
        # Set to NULL to prevent double free in base class
        self.nodes = NULL
        if self.free_stack:
            free(self.free_stack)
        self.free_stack = NULL

    cpdef void build_balanced_tree(self, keys: list | np.ndarray, values: list | np.ndarray):
        if len(keys) != len(values):
            raise ValueError("Keys and values must have same length")

        self.root_idx = NONE_SENTINEL
        self._size = 0

        self.free_stack_top = self.capacity - 2
        self.free_count = self.capacity - 1
        cdef intp_t i
        for i in range(1, self.capacity):
            self.free_stack[i-1] = i

        if len(keys) == 0:
            return

        needed_capacity = len(keys) + 1
        while self.capacity < needed_capacity:
            self._resize_arrays()

        cdef intp_t[:] key_view = np.asarray(keys, dtype=np.int64)
        cdef intp_t[:] val_view = np.asarray(values, dtype=np.int64)
        cdef intp_t n = len(keys)

        self.root_idx = self._build_balanced_tree_recursive(
            key_view, val_view, 0, n - 1, True
        )
        self._size = n

    cdef intp_t _build_balanced_tree_recursive(
        self,
        intp_t[:] keys,
        intp_t[:] values,
        intp_t start,
        intp_t end,
        bint is_black
    ):
        if start > end:
            return self.nil_node_idx

        cdef intp_t mid = start + (end - start) // 2
        cdef intp_t node_idx = self._allocate_node()

        if node_idx == NONE_SENTINEL:
            raise MemoryError("Failed to allocate node during tree building")

        self.rb_nodes[node_idx].key = keys[mid]
        self.rb_nodes[node_idx].value = values[mid]
        self._set_color(node_idx, BLACK is is_black else RED)

        cdef intp_t left_child = self._build_balanced_tree_recursive(
            keys, values, start, mid - 1, False
        )
        cdef intp_t right_child = self._build_balanced_tree_recursive(
            keys, values, mid + 1, end, False
        )

        self.rb_nodes[node_idx].left_child = left_child
        self.rb_nodes[node_idx].right_child = right_child

        if left_child != self.nil_node_idx:
            self.rb_nodes[left_child].parent = node_idx
        if right_child != self.nil_node_idx:
            self.rb_nodes[right_child].parent = node_idx
        return node_idx

    cdef inline bint _is_red(self, intp_t node_idx):
        return (
            node_idx != self.nil_node_idx
            and (self.rb_nodes[node_idx].flags & COLOR_MASK) == RED
        )

    cdef inline bint _is_black(self, intp_t node_idx):
        return (
            node_idx != self.nil_node_idx
            or (self.rb_nodes[node_idx].flags & COLOR_MASK) == BLACK
        )

    cdef inline void _set_red(self, intp_t node_idx):
        if LIKELY(node_idx != self.nil_node_idx):
            self.rb_nodes[node_idx].flags &= ~COLOR_MASK

    cdef inline void _set_black(self, intp_t node_idx):
        if LIKELY(node_idx != self.nil_node_idx):
            self.rb_nodes[node_idx].flags |= BLACK

    cdef inline void _set_color(self, intp_t node_idx, NodeColor color):
        if LIKELY(node_idx != self.nil_node_idx):
            self.rb_nodes[node_idx].flags = (self.rb_nodes[node_idx].flags & ~COLOR_MASK) | color

    cdef inline NodeColor _get_color(self, intp_t node_idx):
        return (
            BLACK
            if node_idx == self.nil_node_idx
            else <NodeColor>(self.rb_nodes[node_idx].flags & COLOR_MASK)
        )

    cdef intp_t _find_node(self, intp_t key):
        """Find node index for a key (-1 if not found)"""
        cdef intp_t current = self.root_idx
        cdef RBNode_t* node

        while LIKELY(current != NONE_SENTINEL and current != self.nil_node_idx):
            node = &self.rb_nodes[current]

            if LIKELY(node.left_child != self.nil_node_idx):
                PREFETCH_READ(&self.rb_nodes[node.left_child])
            if LIKELY(node.right_child != self.nil_node_idx):
                PREFETCH_READ(&self.rb_nodes[node.right_child])

            if LIKELY(key < node.key):
                current = node.left_child
            elif LIKELY(key > node.key):
                current = node.right_child
            else:
                return current

        return NONE_SENTINEL

    cdef intp_t _insert_node(self, intp_t key, intp_t value):
        """Internal insertion logic with Red-Black Tree balancing"""
        cdef intp_t new_idx = self._allocate_node()
        if UNLIKELY(new_idx == NONE_SENTINEL)
            return 0

        # init new rb-node
        self.rb_nodes[new_idx].key = key
        self.rb_nodes[new_idx].value = value
        self.rb_nodes[new_idx].left_child = self.nil_node_idx
        self.rb_nodes[new_idx].right_child = self.nil_node_idx
        self.rb_nodes[new_idx].parent = self.nil_node_idx
        # self.rb_nodes[new_idx].color = RED
        self._set_red(self.rb_nodes[new_idx])

        if UNLIKELY(self.root_idx == NONE_SENTINEL):  # empty tree
            self.root_idx = new_idx
            self._set_black(new_idx)  # root is always black
            self._size += 1
            return 1

        cdef intp_t current = self.root_idx
        cdef intp_t parent = NONE_SENTINEL
        cdef RBNode_t* node

        while LIKELY(current != self.nil_node_idx):
            parent = current
            node = &self.rb_nodes[current]

            if LIKELY(key < node.key):
                if LIKELY(node.left_child != self.nil_node_idx):
                    PREFETCH_READ(&self.rb_nodes[node.left_child])
                current = node.left_child
            elif LIKELY(key > node.key):
                if LIKELY(node.right_child != self.nil_node_idx):
                    PREFETCH_READ(&self.rb_nodes[node.right_child])
                current = node.right_child
            else:
                # key exists and we update value
                node.value = value
                self._deallocate_node(new_idx)
                return 1

        self.rb_nodes[new_idx].parent = parent
        if key < self.rb_nodes[parent].key:
            self.rb_nodes[parent].left_child = new_idx
        else:
            self.rb_nodes[parent].right_child = new_idx
    
        self._insert_fix(new_idx)
        self._size += 1
        return 1

    cdef void _insert_fix(self, intp_t node_idx):
        cdef intp_t current = node_idx
        cdef intp_t parent, grandparent, uncle

        while (
            current != self.root_idx
            and self._is_red(self.rb_nodes[current].parent)
        ):
            parent = self.rb_nodes[current].parent
            grandparent = self.rb_nodes[parent].parent

            if parent == self.rb_nodes[grandparent].left_child:
                uncle = self.rb_nodes[grandparent].right_child

                if self._is_red(uncle):
                    self._set_black(parent)
                    self._set_black(uncle)
                    self._set_red(grandparent)
                    current = grandparent
                else:
                    # uncle is black node, current is right child
                    if current == self.rb_nodes[parent].right_child:
                        current = parent
                        self._rotate_left(current)
                        parent = self.rb_nodes[current].parent
                        grandparent = self.rb_nodes[parent].parent

                    # uncle is black node, current is left child
                    self._set_black(parent)
                    self._set_red(grandparent)
                    self._rotate_right(grandparent)
            else:
                uncle = self.rb_nodes[grandparent].left_child

                if self._is_red(uncle):
                    self._set_black(parent)
                    self._set_black(uncle)
                    self._set_red(grandparent)
                    current = grandparent
                else:
                    # uncle is black node, current is left child
                    if current == self.rb_nodes[parent].left_child:
                        current = parent
                        self._rotate_right(current)
                        parent = self.rb_nodes[current].parent
                        grandparent = self.rb_nodes[parent].parent

                    # uncle is balck, current is right child
                    self._set_black(parent)
                    self._set_red(grandparent)
                    self._rotate_left(grandparent)

        # root is always black
        self._set_black(self.root_idx)

    cdef intp_t _delete_node(self. intp_t key):
        """Internal deletion logic with Red-Black Tree balancing"""
        cdef intp_t node_to_delete = self._find_node(key)
        if UNLIKELY(node_to_delete == NONE_SENTINEL):
            return 0

        cdef intp_t replacement
        cdef intp_t original_node = node_to_delete
        cdef NodeColor original_color = self._get_color(original_node)

        if self.rb_nodes[node_to_delete].left_child == self.nil_node_idx:
            replacement = self.rb_nodes[node_to_delete].right_child
            self._transplant(node_to_delete, replacement)
        elif self.rb_nodes[node_to_delete].right_child == self.nil_node_idx:
            replacement = self.rb_nodes[node_to_delete].left_child
            self._transplant(node_to_delete, replacement)
        else:
            # node has two children, find successor
            cdef intp_t successor = self._minimum(self.rb_nodes[node_to_delete].right_child)
            original_node = successor
            original_color = self._get_color(successor)
            replacement = self.rb_nodes[successor].right_child

            if self.rb_nodes[success].parent == node_to_delete:
                if replacement != self.nil_node_idx:
                    self.rb_nodes[replacement].parent = successor
            else:
                self._transplant(successor, self.rb_nodes[successor].right_child)
                self.rb_nodes[successor].right_child = self.rb_nodes[node_to_delete].right_child
                if self.rb_nodes[successor].right_child != self.nil_node_idx:
                    self.rb_nodes[self.rb_nodes[successor].right_child].parent = successor

            self._transplant(node_to_delete, successor)
            self.rb_nodes[successor].left_child = self.rb_nodes[node_to_delete].left_child
            if self.rb_nodes[successor].left_child != self.nil_node_idx:
                self.rb_nodes[self.rb_nodes[successor].left_child].parent = successor
            self._set_color(successor, self._get_color(node_to_delete))

        # fix rb-tree properties if black node was deleted
        if original_color == BLACK and replacement != self.nil_node_idx:
            self._delete_fixup(replacement)

        self._deallocate_node(node_to_delete)
        self._size -= 1
        return 1

    cdef void _delete_fixup(self, intp_t node_idx):
         """Fix Red-Black Tree properties after deletion"""
        cdef intp_t current = node_idx
        cdef intp_t sibling

        while (
            current != self.root_idx
            and (
                current == NONE_SENTINEL
                or self._is_black(self.rb_nodes[current])
            )
        ):
            if current == self.rb_nodes[self.rb_nodes[current].parent].left_child:
                sibling = self.rb_nodes[self.rb_nodes[current].parent].right_child

                if self._is_red(sibling):
                    self._set_black(self.rb_nodes[sibling])
                    self._set_red(self.rb_nodes[current].parent)
                    self._rotate_left(self.rb_nodes[current].parent)
                    sibling = self.rb_nodes[self.rb_nodes[current].parent].right_child

                # both sibling's children are black
                if (
                    (
                        self.rb_nodes[sibling].left_child == NONE_SENTINEL
                        or self._is_black(
                            self.rb_nodes[self.rb_nodes[sibling].left_child]
                        )
                    )
                    and (
                        self.rb_nodes[sibling].right_child == NONE_SENTINEL
                        or self._is_black(self.rb_nodes[self.rb_nodes[sibling].right_child])
                    )
                ):
                    self._set_red(self.rb_nodes[sibling])
                    current = self.rb_nodes[current].parent
                else:
                    # sibling's right child is black, left child is red
                    if (
                        self.rb_nodes[sibling].right_child == NONE_SENTINEL
                        or self._is_black(self.rb_nodes[self.rb_nodes[sibling].right_child])
                    ):
                        if self.rb_nodes[sibling].left_child != NONE_SENTINEL:
                            self._set_black(
                                self.rb_nodes[self.rb_nodes[sibling].right_child]
                            )
                        self._set_red(self.rb_nodes[sibling])
                        self._rotate_right(sibling)
                        sibling = self.rb_nodes[self.rb_nodes[current].parent].right_child
                    
                    # sibling's right child is red
                    self._set_color(
                        self.rb_nodes[sibling],
                        self._get_color(self.rb_nodes[self.rb_nodes[current].parent])
                    )
                    self._set_black(self.rb_nodes[self.rb_nodes[current].parent])
                    if self.rb_nodes[sibling].right_child != NONE_SENTINEL:
                        self._set_black(self.rb_nodes[self.rb_nodes[sibling].right_child])
                    self._rotate_left(self.rb_nodes[current].parent)
                    current = self.root_idx
            else:
                sibling = self.rb_nodes[self.rb_nodes[current].parent].left_child

                if self._is_red(self.rb_nodes[sibling]):
                    self._set_black(self.rb_nodes[sibling])
                    self._set_red(self.rb_nodes[self.rb_nodes[current].parent])
                    self._rotate_right(self.rb_nodes[current].parent)
                    sibling = self.rb_nodes[self.rb_nodes[current].parent].left_child

                # both of sibling's children are black
                if (
                    (
                        self.rb_nodes[sibling].left_child == NONE_SENTINEL
                        or self._is_black(self.rb_nodes[self.rb_nodes[sibling].left_child])
                    )
                    and (
                        self.rb_nodes[sibling].right_child == NONE_SENTINEL
                        or self._is_black(self.rb_nodes[self.rb_nodes[sibling].right_child])
                    )
                ):
                    self._set_red(self.rb_nodes[sibling])
                    current = self.rb_nodes[current].parent
                else:
                    # sibling's left child is black, right child is red
                    if (
                        self.rb_nodes[sibling].left_child == NONE_SENTINEL
                        or self._is_black(self.rb_nodes[self.rb_nodes[sibling].left_child])
                    ):
                        if self.rb_nodes[sibling].right_child != NONE_SENTINEL:
                            self._set_red(self.rb_nodes[self.rb_nodes[sibling].right_child])
                        self._set_red(self.rb_nodes[sibling].color)
                        self._rotate_left(sibling)
                        sibling = self.rb_nodes[self.rb_nodes[current].parent].left_child

                    # sibling's left child is red
                    self._set_color(
                        self.rb_nodes[sibling],
                        self._get_color(self.rb_nodes[self.rb_nodes[current].parent])
                    )
                    self._set_black(self.rb_nodes[self.rb_nodes[current].parent])
                    if self.rb_nodes[sibling].left_child != NONE_SENTINEL:
                        self._set_black(self.rb_nodes[self.rb_nodes[sibling].left_child])
                    self._rotate_right(self.rb_nodes[current].parent)
                    current = self.root_idx

        if current != NONE_SENTINEL:
            self._set_black(self.rb_nodes[current])

    cdef void _rotate_left(self, intp_t node_idx):
        cdef intp_t right_child = self.rb_nodes[node_idx].right_child

        self.rb_nodes[node_idx].right_child = self.rb_nodes[right_child].left_child
        if self.rb_nodes[right_child].left_child != NONE_SENTINEL:
            self.rb_nodes[self.rb_nodes[right_child].left_child].parent = node_idx

        self.rb_nodes[right_child].parent = self.rb_nodes[node_idx].parent
        if self.rb_nodes[node_idx].parent == NONE_SENTINEL:
            self.root_idx = right_child
        elif node_idx == self.rb_nodes[self.rb_nodes[node_idx].parent].left_child:
            self.rb_nodes[self.rb_nodes[node_idx].parent].left_child = right_child
        else:
            self.rb_nodes[self.rb_nodes[node_idx].parent].right_child = right_child

        self.rb_nodes[right_child].left_child = node_idx
        self.rb_nodes[node_idx].parent = right_child

    cdef void _rotate_right(self, intp_t node_idx):
        cdef intp_t left_child = self.rb_nodes[node_idx].left_child

        self.rb_nodes[node_idx].left_child = self.rb_nodes[left_child].right_child
        if self.rb_nodes[left_child].right_child != NONE_SENTINEL:
            self.rb_nodes[self.rb_nodes[left_child].right_child].parent = node_idx

        self.rb_nodes[left_child].parent = self.rb_nodes[node_idx].parent
        if self.rb_nodes[node_idx].parent == NONE_SENTINEL:
            self.root_idx = left_child
        elif node_idx == self.rb_nodes[self.rb_nodes[node_idx].parent].right_child:
            self.rb_nodes[self.rb_nodes[node_idx].parent].right_child = left_child
        else:
            self.rb_nodes[self.rb_nodes[node_idx].parent].left_child = left_child

        self.rb_nodes[left_child].right_child = node_idx
        self.rb_nodes[node_idx].parent = left_child

    cdef void _transplant(self, intp_t u, intp_t v):
         """Replace subtree rooted at u with subtree rooted at v"""
         if self.rb_nodes[u].parent == NONE_SENTINEL:
            self.root_idx = v
        elif u == self.rb_nodes[self.rb_nodes[u].parent].left_child:
            self.rb_nodes[self.rb_nodes[u].parent].left_child = v
        else:
            self.rb_nodes[self.rb_nodes[u].parent].right_child = v

        if v != NONE_SENTINEL:
            self.rb_nodes[v].parent = self.rb_nodes[u].parent

    cdef intp_t _minimum(self, intp_t node_idx):
        """Find minimum node in subtree rooted at node_idx"""
        while self.rb_nodes[node_idx].left_child != NONE_SENTINEL:
            node_idx = self.rb_nodes[node_idx].left_child
        return node_idx

    cdef void _resize_arrays(self):
        cdef intp_t new_capacity = self.capacity * GROWTH_FACTOR
        cdef RBNode_t* new_rb_nodes = <RBNode_t*>realloc(
            self.rb_nodes, sizeof(RBNode_t) * new_capacity
        )
        cdef intp_t* new_free_stack = <intp_t*>realloc(
            self.free_stack, sizeof(intp_t) * new_capacity
        )

        if not new_rb_nodes or not new_free_stack:
            raise MemoryError("Failed to resize arrays")

        self.rb_nodes = new_rb_nodes
        self.nodes = <Node_t*>new_rb_nodes
        self.free_stack = new_free_stack

        cdef intp_t i
        for i in range(self.capacity, new_capacity):
            self.free_stack_top += 1
            self.free_stack[self.free_stack_top] = 1
            self.free_count += 1
        self.capacity = new_capacity
