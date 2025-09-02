from typing import Any

from src import DWayHeap


def get_topk(heap: DWayHeap, k: int) -> list[Any]:
    """
    Function to get the top-K elements from a heap.

    In case of a max heap, the top K elements with the highest priority will
    be retrieved; for a min heap the top K elements with the lowest priority
    will be retrieved.

    Parameters
    ----------
    heap : DWayHeap
        A DWayHeap object
    k : int
        The number of 'top-K' elements to retrieve.

    Returns
    -------
    list[Any]
        The 'top-K' elements.
    """
    if k <= 0:
        return []
    if heap.is_empty():
        return []

    pairs = [(p, e) for p, e in heap._pairs]
    if heap.is_max_heap:
        pairs.sort(key=lambda x: x[0], reverse=True)
    else:
        pairs.sort(key=lambda x: x[0])
    return [e for _, e in pairs[:k]]
