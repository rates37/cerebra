# cerebra

A 'from-scratch' Python Neural Network library implementing automatic gradient tracking via computational graph.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This is a minimal, "from-scratch" Pytorch-like library, built with Python nad NumPy. It provides:

* A dynamic computational graph with automatic gradient accumulation and computation
* A small Pytorch-like neural network API
* Core layers including Fully Connected layers and Conv2D layers
* Common Activation functions, Loss functions, and optimisers


## How it works

### Computational Graph Core

#### `Node` class:

This represents a tensor in the graph, holding:

* `value`: A NumPy array
* `grad`: The gradient of the value with respect to some final loss
* `parents`: Upstream nodes that the output value of this node relies on
* `op`: The operation that created this Node.

#### `Variable` class:

A leaf node (has no `parents` and no `op`). Typically used to represent inputs or trainable parameters.


#### `Operation` (Abstract) class:

Defines two methods:

* `forward(input)` -> `output`
* `backward(output_gradients, node)` -> `[input_gradients]`

Child classes (Add, Mul, MatMul, etc.) implement Operation methods.


#### Computational Graph Explained

In a fully explicit graph, there would be 'Value Nodes' (holding tensors) and 'Operation Nodes' (implementing the logic of the operation). Sequences of edges would go Value -> Operation -> Value ... 

In this implementation, the graph is compressed, and operation node is collapsed into the child Value Node, keeping it as the `child.op` attribute, and `child.parents` are the upstream nodes.

Conceptually, this implementation can be thought of as following the rules:

* Nodes carry data
* Operations are transformations on edges between nodes.

The directed edge from parent to child contains not just a link, but also the operation by which  the parent's values combine to create the child node's value. When we write the operation $z = x + y$, we create a new Node $z$, with parents $[x, y]$, and store an `Add` operation in the $z$ node's `op` attribute.

When `loss.backward()` is invoked, the following occurs:

1. The Value nodes in the graph are topologically sorted. This is to ensure the child gradients required to calculate the gradient of a parent node are always available in time.

2. All node's gradients are zeroed/cleared.

3. Set the output node's gradient to 1 (as $\partial \mathcal{L} / \partial \mathcal{L} = 1$).

4. Walk the topologically-sorted nodes in reverse:
    * At each Value Node, `n`, inspect `n.op`:
    * Call `n.op.backward(n.grad, n)`, which will return a list of gradients (corresponding to each parent node)
    * Accumulate each computed gradient into the corresponding parent's `grad` attribute.


### Neural Network API

<!-- todo: -->

## Requirements
* Python >= 3.12

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rates37/cerebra.git
```

2. Navigate to main project directory:
```bash
cd cerebra
```

3. Setup a Virtual Environment (optional)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install package locally:
```bash
pip install -e .
```

5. Verify installation:
If no error occur, the package has been installed successfully.
```bash
python -c "import cerebra"
```

### Uninstallation
To remove the package:
```bash
pip uninstall cerebra
```

## Quick Start

Coming soon. For now, see the `examples` directory.

## License

MIT License - See LICENSE file for details.

## Contributions

Contributions are always appreciated. Please feel free to fork this repo, add your features/bug fixes, and open a pull request. If contributing a new feature, be sure to add sufficient tests to ensure the correctness of the feature.
