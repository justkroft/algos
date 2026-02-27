# Data Structures and Algorithms in Cython
This repository provides high-performance implementations of several classic and advanced data structures, written in Cython for speed and efficiency.
The aim is to combine the ease of use of Python with the performance of C-level code.

## Implemented Data Structures
### D-Way Heap
A generalization of the binary heap where each node can have up to d children.
- Efficient for priority queues with customizable branching factor
- Supports insert, extract-min/max, and decrease-key operations

### Binary Search Tree (BST)
A node-based data structure where each node has at most two children, with left keys smaller and right keys larger.
- Search, insert, and delete operations in average-case O(log n)
- Forms the basis for many advanced tree structures

### Red-Black Tree (RBT)
A self-balancing binary search tree with extra color properties to maintain balance
- Guarantees logarithmic height: `h ≤ 2 log₂(n+1)`
- Ensures efficient worst-case performance for search, insert, and delete


### Treap
A randomized balanced binary search tree combining a BST (by keys) and a heap (by priorities).
- Expected O(log n) complexity for search, insert, and delete

## Testing
Tests are written with `pytest` and live under the `tests/` directory. Run all tests with the followign command:
```bash
uv run pytest
```

## Time Complexity
| Data Structure           | Search             | Insert             | Delete             | Notes                                                                 |
| ------------------------ | ------------------ | ------------------ | ------------------ | --------------------------------------------------------------------- |
| D-Way Heap               | O(d · log(d) n)    | O(log(d) n)        | O(log(d) n)        | Larger *d* reduces height but increases percolation cost              |
| Binary Search Tree (BST) | O(h), avg O(log n) | O(h), avg O(log n) | O(h), avg O(log n) | `h` = tree height; worst-case O(n) if unbalanced                      |
| Red-Black Tree (RBT)     | O(log n)           | O(log n)           | O(log n)           | Self-balancing BST; uses rotations and recoloring to maintain balance |
| Treap                    | O(log n) expected  | O(log n) expected  | O(log n) expected  | Balancing is randomized with heap priorities                          |


# Installation
*Prerequisites — you need CMake and a C compiler installed on your machine.*

Clone the repository
```bash
git clone https://github.com/justkroft/algos.git
cd algos
```

and setup your virtual environment using `uv`.

```bash
uv venv  # create environment
uv sync  # sync all dependencies from toml file
```

Then, activate the environment:

```bash
source .venv/bin/activate  # on Mac
./venv/Scripts/activate    # on Windows
```

Once the environment is setup and all the dependencies are installed, you can create a local, editable install:

```bash
uv pip install --no-build-isolation -e .
```

# Contributing
Contributions are welcome.\
Open an issue for bug reports or feature requests, or submit a pull request with improvements.
