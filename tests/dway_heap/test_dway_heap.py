import pytest

from src.dway_heap.dway_heap import DWayHeap


class TestDWayHeap:
    def test_empty_head(self):
        """Test empty heap creation and basic properties"""
        heap = DWayHeap()
        assert len(heap) == 0
        assert heap.is_empty()

        with pytest.raises(RuntimeError):
            heap.peek()
        with pytest.raises(RuntimeError):
            heap.top()

    def test_max_heap_basic(self):
        """Test basic max heap operations"""
        elements = ["A", "B", "C", "D"]
        priorities = [10.0, 5.0, 15.0, 3.0]

        heap = DWayHeap.max_heap(elements, priorities)

        assert len(heap) == 4
        assert not heap.is_empty()
        assert heap.peek() == "C"
        assert heap._validate()


        assert heap.top() == "C"
        assert heap.top() == "A"
        assert heap.top() == "B"
        assert heap.top() == "D"
        assert heap.is_empty()

    def test_min_heap_basic(self):
        """Test basic min heap operations"""
        elements = ["A", "B", "C", "D"]
        priorities = [10.0, 5.0, 15.0, 3.0]

        heap = DWayHeap.min_heap(elements, priorities)

        assert len(heap) == 4
        assert heap.peek() == "D"
        assert heap._validate()

        assert heap.top() == "D"
        assert heap.top() == "B"
        assert heap.top() == "A"
        assert heap.top() == "C"
        assert heap.is_empty()

    def test_insert_operations(self):
        """Test insert operations"""
        heap = DWayHeap.max_heap()

        heap.insert("first", 10.0)
        assert heap.peek() == "first"
        assert len(heap) == 1

        heap.insert("second", 20.0)
        assert heap.peek() == "second"
        assert len(heap) == 2

        heap.insert("third", 5.0)
        assert heap.peek() == "second"
        assert len(heap) == 3
        assert heap._validate()

    @pytest.mark.parametrize("branching_factor", [2, 3, 4, 8])
    def test_different_branching_factors(self, branching_factor):
        """Test heap with different branching factors"""
        elements = [f"elem_{i}" for i in range(10)]
        priorities = list(range(10, 0, -1))  # 10, 9, 8, ..., 1

        heap = DWayHeap.max_heap(elements, priorities, branching_factor)
        assert heap.branching_factor == branching_factor
        assert heap._validate()

        prev_priority = float('inf')
        while not heap.is_empty():
            elem = heap.top()
            current_priority = priorities[elements.index(elem)]
            assert current_priority <= prev_priority
            prev_priority = current_priority

    def test_mixed_data_types(self):
        """Test heap with mixed element types"""
        elements = ["string", 42, [1, 2, 3], {"key": "value"}]
        priorities = [1.0, 4.0, 2.0, 3.0]

        heap = DWayHeap.max_heap(elements, priorities)
        assert heap._validate()

        # Should extract in priority order: 42, {}, [1,2,3], "string"
        assert heap.top() == 42
        assert heap.top() == {"key": "value"}
        assert heap.top() == [1, 2, 3]
        assert heap.top() == "string"

    def test_duplicate_priorities(self):
        """Test heap with duplicate priorities"""
        elements = ["A", "B", "C", "D"]
        priorities = [10.0, 10.0, 5.0, 10.0]

        heap = DWayHeap.max_heap(elements, priorities)
        assert heap._validate()

        # Extract all elements with priority 10.0 first
        high_priority_elements = []
        for _ in range(3):
            elem = heap.top()
            high_priority_elements.append(elem)

        assert set(high_priority_elements) == {"A", "B", "D"}
        assert heap.top() == "C"

    def test_error_conditions(self):
        """Test error conditions"""
        with pytest.raises(ValueError):
            DWayHeap(["A", "B"], [1.0])

        # Invalid branching factor
        with pytest.raises(ValueError):
            DWayHeap([], [], branching_factor=1)

        with pytest.raises(ValueError):
            DWayHeap([], [], branching_factor=0)

    def test_heap_validation(self):
        """Test heap property validation"""
        heap = DWayHeap.max_heap(["A", "B", "C"], [3.0, 1.0, 2.0])
        assert heap._validate()

        heap.insert("D", 4.0)
        assert heap._validate()
