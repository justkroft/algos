from src.dway_heap import DWayHeap


priorities = [10.5, 3.2, 15.0, 7.8, 20.1, 1.5]
elements = ["low", "very_low", "medium", "low_med", "high", "lowest"]

# Create a binary heap (branching_factor=2)
print("Creating binary heap...")
heap = DWayHeap(elements, priorities, branching_factor=2)

# Test basic properties
print(f"Heap size: {len(heap)}")
print(f"Is empty: {heap.is_empty()}")
print(f"First leaf index: {heap.first_leaf_index()}")
