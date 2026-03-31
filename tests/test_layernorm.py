import unittest
import numpy as np
from typing import Callable
from cerebra import Node, LayerNorm, LayerNormOp

EPSILON = 1e-6

def numerical_gradient(func: Callable[[], Node], node: Node, h=1e-6) -> np.ndarray:
    input_value = node.value.copy()
    grad = np.zeros_like(input_value, dtype=np.float64)

    for i in np.ndindex(input_value.shape):
        val = input_value[i]

        # calculate f(x+h):
        x_plus_h = input_value.copy()
        x_plus_h[i] = val + h
        node.value = x_plus_h
        f_x_plus_h = func().value.item()

        # calculate f(x-h):
        x_minus_h = input_value.copy()
        x_minus_h[i] = val - h
        node.value = x_minus_h
        f_x_minus_h = func().value.item()

        # central difference method:
        grad[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

    # restore input value of node
    node.value = input_value
    return grad

class TestLayerNorm(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(42)

    def test_ln_forward(self):
        N, D = 4, 3
        x = self.default_rng.random((N, D))
        ln = LayerNorm(D)
        
        y = ln(Node(x))
        
        # Normalise over D
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected_y = (x - mean) / np.sqrt(var + ln.eps)
        
        self.assertTrue(np.allclose(y.value, expected_y, atol=EPSILON))

    def test_ln_backward(self):
        N, D = 4, 3
        ln_op = LayerNormOp()
        x_val = self.default_rng.random((N, D))
        gamma_val = self.default_rng.random(D)
        beta_val = self.default_rng.random(D)
        
        x_node = Node(x_val)
        gamma_node = Node(gamma_val)
        beta_node = Node(beta_val)
        
        y_val = ln_op.forward(x_val, gamma_val, beta_val)
        grad_output = self.default_rng.random((N, D))
        
        dummy_output = Node(y_val, parents=[x_node, gamma_node, beta_node], op=ln_op)
        grads = ln_op.backward(grad_output, dummy_output)
        
        def get_loss():
            out = ln_op.forward(x_node.value, gamma_node.value, beta_node.value)
            return Node(np.sum(out * grad_output))
            
        dx_num = numerical_gradient(get_loss, x_node)
        dg_num = numerical_gradient(get_loss, gamma_node)
        db_num = numerical_gradient(get_loss, beta_node)
        
        self.assertTrue(np.allclose(grads[0], dx_num, atol=1e-5))
        self.assertTrue(np.allclose(grads[1], dg_num, atol=1e-5))
        self.assertTrue(np.allclose(grads[2], db_num, atol=1e-5))

    def test_ln_zero_variance(self):
        # All features are the same per sample -> zero variance
        N, D = 2, 3
        x = Node(np.array([[5.0, 5.0, 5.0], [2.0, 2.0, 2.0]]))
        ln = LayerNorm(D)
        y = ln(x)

        self.assertTrue(np.allclose(y.value, np.zeros((N, D)), atol=EPSILON))

    def test_ln_d1(self):
        # D=1 case: LayerNorm on single feature always results in 0
        N, D = 3, 1
        x = Node(np.array([[1.0], [2.0], [3.0]]))
        ln = LayerNorm(D)
        y = ln(x)
        self.assertTrue(np.allclose(y.value, np.zeros((N, D)), atol=EPSILON))
        
    def test_ln_large_values(self):
        N, D = 2, 2
        x = Node(np.array([[1e5, -1e5], [1e6, 0.0]]))
        ln = LayerNorm(D)
        op = LayerNormOp()
        out_val = op.forward(x.value, ln.gamma.value, ln.beta.value)
        y = Node(out_val, parents=[x, ln.gamma, ln.beta], op=op)
        
        self.assertTrue(np.all(np.isfinite(y.value)))

        grad_out = np.ones((N, D))
        grads = op.backward(grad_out, y)
        self.assertTrue(np.all(np.isfinite(grads[0])))
