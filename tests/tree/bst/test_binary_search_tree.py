import numpy as np
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

    def test_deletion_one_child(self):
        self.bst.insert(50, 500)
        self.bst.insert(30, 300)
        self.bst.insert(20, 200)  # Left child of 30

        assert self.bst.delete(30)
        assert len(self.bst) == 2
        assert 30 not in self.bst
        assert 50 in self.bst
        assert 20 in self.bst
        assert self.bst[20] == 200

    def test_deletion_two_children(self):
        keys_values = [
            (50, 500),
            (30, 300),
            (70, 700),
            (20, 200),
            (40, 400),
            (60, 600),
            (80, 800)
        ]
        for k, v in keys_values:
            self.bst.insert(k, v)

        assert self.bst.delete(50)
        assert len(self.bst) == 6
        assert 50 not in self.bst

        # Verify tree is still valid
        remaining_keys = [30, 70, 20, 40, 60, 80]
        for key in remaining_keys:
            assert key in self.bst

    def test_deletion_root_node(self):
        # Single node tree
        self.bst.insert(42, 420)
        assert self.bst.delete(42)
        assert len(self.bst) == 0
        assert self.bst.is_empty()

        self.bst.insert(50, 500)
        self.bst.insert(30, 300)
        assert self.bst.delete(50)
        assert len(self.bst) == 1
        assert self.bst[30] == 300

    def test_memory_efficiency(self):
        initial_capacity = 64

        # Fill beyond initial capacity to trigger resize
        for i in range(initial_capacity + 10):
            self.bst.insert(i, i * 2)

        # Should have grown capacity
        assert len(self.bst) == initial_capacity + 10

        # Delete many elements - free nodes should be reused
        for i in range(0, initial_capacity, 2):
            self.bst.delete(i)

        # Insert new elements - should reuse freed nodes
        for i in range(1000, 1005):
            self.bst.insert(i, i)

        # Tree should still work correctly
        assert 1000 in self.bst
        assert self.bst[1000] == 1000

    def test_edge_cases(self):
        self.bst.insert(-50, 50)
        self.bst.insert(-100, 100)
        self.bst.insert(0, 0)

        assert self.bst[-50] == 50
        assert self.bst[-100] == 100
        assert self.bst[0] == 0

        # Test with large numbers
        large_key = 2**31 - 1
        self.bst.insert(large_key, large_key)
        assert self.bst[large_key] == large_key

    def test_traversal(self):
        keys = [50, 30, 70, 20, 40, 60, 80]
        values = [100, 200, 300, 400, 500, 600, 700]
        self.bst.build_tree(keys, values)

        # inorder test
        np.testing.assert_array_equal(
            np.array(self.bst.inorder()),
            np.array([20, 30, 40, 50, 60, 70, 80])
        )

        # preorder test
        np.testing.assert_array_equal(
            np.array(self.bst.preorder()),
            np.array([50, 30, 20, 40, 70, 60, 80])
        )

        # postorder test
        np.testing.assert_array_equal(
            np.array(self.bst.postorder()),
            np.array([20, 40, 30, 60, 80, 70, 50])
        )

    def test_dict_api(self):
        keys = [50, 30, 70, 20, 40, 60, 80]
        values = [100, 200, 300, 400, 500, 600, 700]
        self.bst.build_tree(keys, values)

        # test keys
        np.testing.assert_array_equal(
            self.bst.keys(),
            np.array([20, 30, 40, 50, 60, 70, 80])
        )

        # test values
        np.testing.assert_array_equal(
            self.bst.values(),
            np.array([400, 200, 500, 100, 600, 300, 700])
        )

        # test items
        np.testing.assert_array_equal(
            np.array(self.bst.items()),
            np.array(
                [
                    (20, 400),
                    (30, 200),
                    (40, 500),
                    (50, 100),
                    (60, 600),
                    (70, 300),
                    (80, 700)
                ]
            )
        )

    def test_multiple(self):
        keys = [50, 30, 70, 20, 40, 60, 80]
        values = [100, 200, 300, 400, 500, 600, 700]
        self.bst.build_tree(keys, values)

        np.testing.assert_array_equal(
            self.bst.get_multiple([20, 30, 80]),
            np.array([400, 200, 700])
        )

        np.testing.assert_array_equal(
            self.bst.contains_multiple([10, 20, 30, 90]),
            np.array([False, True, True, False])
        )

        self.bst.delete_multiple([20, 30])
        assert 20 not in self.bst
        assert 30 not in self.bst
