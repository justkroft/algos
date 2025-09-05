import numpy as np
import pytest

from src.treap.randomized_treap import RandomizedTreap


class TestRandomizedTreap:
    def test_empty_treap(self):
        treap = RandomizedTreap()
        assert len(treap) == 0
        assert treap.is_empty()

        with pytest.raises(RuntimeError):
            treap.top()

        with pytest.raises(RuntimeError):
            treap.peek()

        assert treap.search(1.0) is None

        with pytest.raises(IndexError):
            treap.select(0)

    def test_single_element(self):
        treap = RandomizedTreap()
        treap.insert(5.0, "five")

        assert len(treap) == 1
        assert not treap.is_empty()
        assert treap.peek() == 5.0
        assert treap.search(5.0) == "five"
        assert treap.rank(5.0) == 0
        assert treap.select(0) == "five"

        # Test top() removes element
        assert treap.top() == "five"
        assert len(treap) == 0
        assert treap.is_empty()

    def test_basic_operations(self):
        treap = RandomizedTreap()

        data = [(3.0, "three"), (1.0, "one"), (4.0, "four"), (2.0, "two")]
        for key, value in data:
            treap.insert(key, value)

        assert len(treap) == 4

        assert treap.search(1.0) == "one"
        assert treap.search(2.0) == "two"
        assert treap.search(3.0) == "three"
        assert treap.search(4.0) == "four"
        assert treap.search(5.0) is None

        assert treap.peek() == 1.0

        treap.remove(2.0)
        assert len(treap) == 3
        assert treap.search(2.0) is None

        with pytest.raises(KeyError):
            treap.remove(10.0)

    def test_rank_operations(self):
        """Test rank and select operations"""
        treap = RandomizedTreap()
        keys = [3.0, 1.0, 4.0, 1.5, 2.0]
        values = ["three", "one", "four", "one_five", "two"]

        for key, value in zip(keys, values):
            treap.insert(key, value)

        # ranks 0-based, sorted order: 1.0, 1.5, 2.0, 3.0, 4.0
        assert treap.rank(1.0) == 0
        assert treap.rank(1.5) == 1
        assert treap.rank(2.0) == 2
        assert treap.rank(3.0) == 3
        assert treap.rank(4.0) == 4
        assert treap.rank(0.5) == -1  # Not found

        # select
        assert treap.select(0) == "one"
        assert treap.select(1) == "one_five"
        assert treap.select(2) == "two"
        assert treap.select(3) == "three"
        assert treap.select(4) == "four"

        with pytest.raises(IndexError):
            treap.select(5)
        with pytest.raises(IndexError):
            treap.select(-1)

    def test_duplicate_keys(self):
        treap = RandomizedTreap()

        treap.insert(2.0, "first")
        treap.insert(2.0, "second")  # Duplicate key
        treap.insert(2.0, "third")  # Another duplicate

        assert len(treap) == 3
        # Since keys are equal, any of the values could be returned
        result = treap.search(2.0)
        assert result in ["first", "second", "third"]

        # All should have same rank
        assert treap.rank(2.0) in [0, 1, 2]  # Could be any due to equal keys

    def test_top_extraction(self):
        treap = RandomizedTreap()
        keys = [3.0, 1.0, 4.0, 2.0]
        values = ["three", "one", "four", "two"]

        for key, value in zip(keys, values):
            treap.insert(key, value)

        extracted = []
        while not treap.is_empty():
            extracted.append(treap.top())

        assert extracted == ["one", "two", "three", "four"]
        assert len(treap) == 0

    def test_large_dataset(self):
        treap = RandomizedTreap()
        n = 1000

        # Insert random data
        np.random.seed(42)
        keys = np.random.uniform(0, 1000, n)

        for i, key in enumerate(keys):
            treap.insert(key, f"value_{i}")

        assert len(treap) == n
        for i in range(0, min(10, n)):
            assert treap.search(keys[i]) == f"value_{i}"

        for _ in range(10):
            rank = np.random.randint(0, len(treap))
            selected_value = treap.select(rank)

            value_idx = int(selected_value.split("_")[1])
            found_rank = treap.rank(keys[value_idx])

            assert found_rank >= 0

    def test_memory_growth(self):
        treap = RandomizedTreap(initial_capacity=4)

        # Insert more than initial capacity
        for i in range(10):
            treap.insert(float(i), f"value_{i}")

        assert len(treap) == 10
        for i in range(10):
            assert treap.search(float(i)) == f"value_{i}"

    def test_edge_case_keys(self):
        treap = RandomizedTreap()

        edge_keys = [0.0, -1.0, float("inf"), -float("inf"), 1e-10, 1e10]

        for i, key in enumerate(edge_keys):
            if not np.isinf(key):
                treap.insert(key, f"value_{i}")

        # Test search for finite values
        finite_keys = [k for k in edge_keys if not np.isinf(k)]
        for i, key in enumerate(finite_keys):
            expected_value = f"value_{edge_keys.index(key)}"
            assert treap.search(key) == expected_value
