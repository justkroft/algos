from src.dway_heap.dway_heap import DWayHeap
from src.dway_heap.topk import get_topk


class TestGetTopK:
    def test_get_topk_with_negative_k(self):
        heap = DWayHeap(["A", "B", "C"], [10, 5, 3])
        result = get_topk(heap, -1)
        assert result == []

    def test_get_topk_with_empty_heap(self):
        heap = DWayHeap()
        result = get_topk(heap, 5)
        assert result == []

    def test_get_topk_max_heap(self):
        heap = DWayHeap(
            ["ten", "five", "fifteen", "one", "twenty"],
            [10, 5, 15, 1, 20],
            is_max_heap=True
        )
        result = get_topk(heap, 3)
        expected = ["twenty", "fifteen", "ten"]
        assert result == expected

    def test_get_topk_min_heap(self):
        heap = DWayHeap(
            ["ten", "five", "fifteen", "one", "twenty"],
            [10, 5, 15, 1, 20],
            is_max_heap=False
        )
        result = get_topk(heap, 3)
        expected = ["one", "five", "ten"]
        assert result == expected

    def test_get_topk_with_duplicate_priorities_max_heap(self):
        heap = DWayHeap(
            ['a', 'b', 'c', 'd'],
            [10, 10, 5, 15],
            is_max_heap=True
        )
        result = get_topk(heap, 3)
        assert len(result) == 3
        assert 'd' in result
        # both priority 10
        assert 'a' in result
        assert 'b' in result
        assert 'c' not in result

    def test_get_topk_with_duplicate_priorities_min_heap(self):
        heap = DWayHeap(
            ['a', 'b', 'c', 'd'],
            [10, 10, 5, 15],
            is_max_heap=False
        )
        result = get_topk(heap, 3)
        assert len(result) == 3
        assert 'c' in result
        # both priority 10
        assert 'a' in result
        assert 'b' in result
        assert 'd' not in result

    def test_get_topk_with_different_data_types(self):
        elements = ["string", 42, [1, 2, 3], {"key": "value"}]
        priorities = [1, 2, 3, 4]
        heap = DWayHeap(elements, priorities)
        result = get_topk(heap, 2)
        expected = [{'key': 'value'}, [1, 2, 3]]
        assert result == expected

    def test_get_topk_with_negative_priorities(self):
        elements = ["neg_ten", "zero", "five", "neg_five"]
        priorities = [-10, 0, 5, -5]

        # test max heap
        max_heap = DWayHeap(elements, priorities, is_max_heap=True)
        result_max = get_topk(max_heap, 2)
        expected_max = ["five", "zero"]
        assert result_max == expected_max

        # test min heap
        min_heap = DWayHeap(elements, priorities, is_max_heap=False)
        result_min = get_topk(min_heap, 2)
        expected_min = ["neg_ten", "neg_five"]
        assert result_min == expected_min
