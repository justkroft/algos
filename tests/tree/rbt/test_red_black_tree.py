import numpy as np
import pytest

from src.tree.rbt.red_black_tree import RedBlackTree


class TestRedBlackTree:

    @pytest.fixture
    def empty_tree(self):
        return RedBlackTree()

    @pytest.fixture
    def simple_tree(self):
        tree = RedBlackTree()
        keys = np.array([20, 25, 30, 35, 40])
        values = np.array([200, 250, 300, 350, 400])
        tree.build_tree(keys, values)
        return tree

    @pytest.fixture
    def complex_tree(self):
        tree = RedBlackTree()
        keys = np.array(
            [50, 25, 75, 15, 35, 60, 85, 10,
             20, 30, 40, 55, 65, 80, 90]
        )
        values = np.array(
            [500, 250, 750, 150, 350, 600, 850, 100,
             200, 300, 400, 550, 650, 800, 900]
        )
        tree.build_tree(keys, values)
        return tree

    def test_empty_tree(self, empty_tree):
        assert len(empty_tree) == 0
        assert empty_tree.is_empty()
        assert not bool(empty_tree)
        assert list(empty_tree.keys()) == []
        assert list(empty_tree.values()) == []
        assert empty_tree.items() == []

    def test_insert_tree(self):
        tree = RedBlackTree()
        tree.insert(42, 420)
        tree.insert(69, 690)
        assert len(tree) == 2
        assert not tree.is_empty()
        assert 42 in tree
        assert tree[69] == 690

    def test_build_simple_tree(self, simple_tree):
        assert len(simple_tree) == 5
        assert not simple_tree.is_empty()

        for key in [20, 25, 30, 35, 40]:
            assert key in simple_tree

        keys = simple_tree.keys()
        expected = np.array([20, 25, 30, 35, 40])
        np.testing.assert_array_equal(keys, expected)

    def test_build_complex_tree(self, complex_tree):
        assert len(complex_tree) == 15

        keys = complex_tree.keys()
        expected = np.array(
            [10, 15, 20, 25, 30, 35, 40, 50,
             55, 60, 65, 75, 80, 85, 90]
        )
        np.testing.assert_array_equal(keys, expected)

    def test_build_tree_mismatched_lengths_raise_error(self, empty_tree):
        with pytest.raises(
            ValueError,
            match="Keys and values must have same length"
        ):
            empty_tree.build_tree(np.array([1, 2, 3]), np.array([10, 20]))

    def test_get_keys(self, simple_tree):
        assert simple_tree.get(20) == 200
        assert simple_tree.get(30) == 300
        assert simple_tree.get(40) == 400
        assert simple_tree[25] == 250
        assert simple_tree[35] == 350

    def test_get_non_existent_key_raise_error(self, simple_tree):
        with pytest.raises(KeyError):
            simple_tree.get(999)
        with pytest.raises(KeyError):
            _ = simple_tree[999]

    def test_contains(self, simple_tree):
        assert 20 in simple_tree
        assert 30 in simple_tree
        assert 999 not in simple_tree
        assert simple_tree.contains(25)
        assert not simple_tree.contains(999)

    def test_insert_and_update(self, empty_tree):
        assert empty_tree.insert(50, 500)
        assert len(empty_tree) == 1
        assert empty_tree[50] == 500

        empty_tree[50] = 999
        assert empty_tree[50] == 999
        assert len(empty_tree) == 1

        empty_tree[25] = 250
        assert len(empty_tree) == 2
        assert empty_tree[25] == 250

    def test_delete(self, simple_tree):
        original_size = len(simple_tree)

        assert simple_tree.delete(30)
        assert len(simple_tree) == original_size - 1
        assert 30 not in simple_tree

        assert not simple_tree.delete(999)
        assert len(simple_tree) == original_size - 1

        del simple_tree[20]
        assert 20 not in simple_tree
        assert len(simple_tree) == original_size - 2

        with pytest.raises(KeyError):
            del simple_tree[999]

    def test_get_multiple(self, complex_tree):
        keys = np.array([10, 30, 50, 70, 90, 999])
        results = complex_tree.get_multiple(keys)

        assert results[0] == 100
        assert results[1] == 300
        assert results[2] == 500
        assert results[3] == -1   # key 70 doesn't exist (NONE_SENTINEL)
        assert results[4] == 900
        assert results[5] == -1   # key 999 doesn't exist

    def test_get_multiple_empty_input(self, complex_tree):
        results = complex_tree.get_multiple(np.array([]))
        assert len(results) == 0

    def test_contains_multiple(self, complex_tree):
        keys = np.array([10, 30, 50, 70, 90, 999])
        results = complex_tree.contains_multiple(keys)

        expected = np.array([True, True, True, False, True, False])
        np.testing.assert_array_equal(results, expected)

    def test_delete_multiple(self, complex_tree):
        original_size = len(complex_tree)
        keys_to_delete = np.array([10, 30, 50, 999])
        deleted_count = complex_tree.delete_multiple(keys_to_delete)

        assert deleted_count == 3
        assert len(complex_tree) == original_size - 3

        for key in [10, 30, 50]:
            assert key not in complex_tree

    def test_inorder_traversal(self, complex_tree):
        inorder = complex_tree.inorder()
        expected = np.array(
            [10, 15, 20, 25, 30, 35, 40,
             50, 55, 60, 65, 75, 80, 85, 90]
        )
        np.testing.assert_array_equal(inorder, expected)

        keys = complex_tree.keys()
        np.testing.assert_array_equal(keys, inorder)

    def test_preorder_traversal(self, complex_tree):
        preorder = complex_tree.preorder()
        assert preorder[0] == 50
        assert len(preorder) == 15

        sorted_preorder = np.sort(preorder)
        expected = np.array(
            [10, 15, 20, 25, 30, 35, 40,
             50, 55, 60, 65, 75, 80, 85, 90]
        )
        np.testing.assert_array_equal(sorted_preorder, expected)

    def test_postorder_traversal(self, complex_tree):
        postorder = complex_tree.postorder()
        assert postorder[-1] == 50
        assert len(postorder) == 15

        sorted_postorder = np.sort(postorder)
        expected = np.array(
            [10, 15, 20, 25, 30, 35, 40,
             50, 55, 60, 65, 75, 80, 85, 90]
            )
        np.testing.assert_array_equal(sorted_postorder, expected)

    def test_values_traversal(self, simple_tree):
        values = simple_tree.values()
        expected = np.array([200, 250, 300, 350, 400])
        np.testing.assert_array_equal(values, expected)

    def test_items_traversal(self, simple_tree):
        items = simple_tree.items()
        expected = [(20, 200), (25, 250), (30, 300), (35, 350), (40, 400)]
        assert items == expected

        inorder_items = simple_tree.inorder_items()
        assert inorder_items == expected

    def test_preorder_items(self, simple_tree):

        items = simple_tree.preorder_items()
        assert len(items) == 5

        items_dict = dict(items)
        assert items_dict[20] == 200
        assert items_dict[30] == 300

    def test_postorder_items(self, simple_tree):

        items = simple_tree.postorder_items()
        assert len(items) == 5

        items_dict = dict(items)
        assert items_dict[25] == 250
        assert items_dict[40] == 400
