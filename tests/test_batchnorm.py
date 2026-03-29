import unittest
import numpy as np
from typing import Callable
from cerebra import Node, BatchNorm, Parameter, no_grad, BatchNormOp

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

class TestBatchNorm(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(42)

    def test_bn_forward_2d(self):
        N, C = 4, 3
        x = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        bn = BatchNorm(C)
        bn.gamma.value = np.ones(C)
        bn.beta.value = np.zeros(C)
        
        y = bn(Node(x))
        
        # Expected mean per channel: [5.5, 6.5, 7.5]
        expected_mean = x.mean(axis=0)
        expected_var = x.var(axis=0)
        expected_y = (x - expected_mean) / np.sqrt(expected_var + bn.eps)
        
        self.assertTrue(np.allclose(y.value, expected_y, atol=EPSILON))
        self.assertTrue(np.allclose(bn.running_mean, expected_mean * 0.1, atol=EPSILON))

    def test_bn_forward_4d(self):
        N, C, H, W = 2, 2, 2, 2
        x = self.default_rng.random((N, C, H, W))
        bn = BatchNorm(C)
        
        y = bn(Node(x))
        
        # Normalise over (N, H, W)
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        expected_y = (x - mean) / np.sqrt(var + bn.eps)
        
        self.assertTrue(np.allclose(y.value, expected_y, atol=EPSILON))

    def test_bn_inference(self):
        C = 2
        bn = BatchNorm(C)
        bn.running_mean = np.array([1.0, 2.0])
        bn.running_var = np.array([0.5, 0.5])
        bn.training = False
        
        x = np.array([[2.0, 3.0], [0.0, 1.0]])
        y = bn(Node(x))
        
        expected_y = (x - bn.running_mean) / np.sqrt(bn.running_var + bn.eps)
        self.assertTrue(np.allclose(y.value, expected_y, atol=EPSILON))
        # In inference, it should be a leaf node (no parents/op)
        self.assertEqual(len(y.parents), 0)
        self.assertIsNone(y.op)

    def _check_bn_backward(self, x_shape):
        C = x_shape[1]
        bn_op = BatchNormOp()
        x_val = self.default_rng.random(x_shape)
        gamma_val = self.default_rng.random(C)
        beta_val = self.default_rng.random(C)
        
        x_node = Node(x_val)
        gamma_node = Node(gamma_val)
        beta_node = Node(beta_val)
        
        y_val = bn_op.forward(x_val, gamma_val, beta_val)
        
        # Random output gradient
        grad_output = self.default_rng.random(x_shape)
        
        dummy_output = Node(y_val, parents=[x_node, gamma_node, beta_node], op=bn_op)
        grads = bn_op.backward(grad_output, dummy_output)
        
        def get_loss():
            out = bn_op.forward(x_node.value, gamma_node.value, beta_node.value)
            return Node(np.sum(out * grad_output))
            
        dx_num = numerical_gradient(get_loss, x_node)
        dg_num = numerical_gradient(get_loss, gamma_node)
        db_num = numerical_gradient(get_loss, beta_node)
        
        self.assertTrue(np.allclose(grads[0], dx_num, atol=1e-5))
        self.assertTrue(np.allclose(grads[1], dg_num, atol=1e-5))
        self.assertTrue(np.allclose(grads[2], db_num, atol=1e-5))

    def test_bn_backward_2d(self):
        self._check_bn_backward((4, 3))

    def test_bn_backward_4d(self):
        self._check_bn_backward((2, 2, 4, 4))

