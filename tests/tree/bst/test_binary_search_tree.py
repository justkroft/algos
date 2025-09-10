# import numpy as np
import pytest

from src.tree.bst.binary_search_tree import BinarySearchTree


class TestBinarySearchTree:

    def setup_method(self):
        self.bst = BinarySearchTree()

    def test_empty_tree(self):
        assert len(self.bst) == 0
        assert self.bst.is_empty()
        assert not bool(self.bst)

        with pytest.raises(KeyError):
            self.bst.get(42)

        with pytest.raises(KeyError):
            _ = self.bst[42]

    def test_insert(self):
        assert self.bst.insert(10, 100)

        assert len(self.bst) == 1
        assert not self.bst.is_empty()
        assert bool(self.bst)

    def test_get(self):
        self.bst.insert(10, 100)

        assert self.bst.get(10) == 100
        assert self.bst[10] == 100

    def test_contains(self):
        self.bst.insert(10, 100)

        assert self.bst.contains(10)
        assert 10 in self.bst

        assert not self.bst.contains(20)
        assert 20 not in self.bst

    def test_build_tree(self):
        test_data = [
            (50, 500),
            (30, 300),
            (70, 700),
            (20, 200),
            (40, 400),
            (60, 600),
            (80, 800)
        ]

        for key, value in test_data:
            assert self.bst.insert(key, value)

        assert len(self.bst) == 7

        # Verify all elements are retrievable
        for key, expected_value in test_data:
            assert self.bst[key] == expected_value
            assert key in self.bst

    def test_deletion_leaf_node(self):
        # Build tree: 50 -> (30, 70)
        self.bst.insert(50, 500)
        self.bst.insert(30, 300)
        self.bst.insert(70, 700)

        # Delete leaf node
        assert self.bst.delete(30)
        assert len(self.bst) == 2
        assert 30 not in self.bst
        assert 50 in self.bst
        assert 70 in self.bst

        # Try to delete non-existent key
        assert not self.bst.delete(999)
