import numpy as np
from typing import Callable
from cerebra import Node

# Common constants for tests
EPSILON = 1e-6


def numerical_gradient(func: Callable[[], Node], node: Node, h=1e-6) -> np.ndarray:
    """Compute gradient of node w.r.t. input using central difference.

    Args:
        func (Callable[[], Node]): Callable that when invoked, re-evaluates computational graph
                                   based on the current state of `node`.
        node (Node): The node whose value attribute will be altered to calculate the
                     gradient.
        h (float, optional): Small perturbation value. Defaults to 1e-6.

    Returns:
        np.ndarray: Numpy array representing the numerically calculated gradient.
                    Of same shape as `node.value`.
    """
    input_value = node.value.copy()
    grad = np.zeros_like(input_value, dtype=np.float64)

    for i in np.ndindex(input_value.shape):
        val = input_value[i]

        # calculate f(x+h):
        x_plus_h = input_value.copy()
        x_plus_h[i] = val + h
        node.value = x_plus_h
        f_val = func().value
        f_x_plus_h = f_val.item() if f_val.size == 1 else f_val.sum()

        # calculate f(x-h):
        x_minus_h = input_value.copy()
        x_minus_h[i] = val - h
        node.value = x_minus_h
        f_val = func().value
        f_x_minus_h = f_val.item() if f_val.size == 1 else f_val.sum()

        # central difference method:
        grad[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

    # restore input value of node
    node.value = input_value
    return grad
