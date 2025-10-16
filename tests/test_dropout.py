import unittest
import numpy as np
from typing import Callable
from cerebra import Node, Parameter, Operation, Dropout

EPSILON = 1e-6


class TestDropout(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

    def tearDown(self) -> None:
        pass

    def test_dropout_forward(self) -> None:
        x_np = np.random.randn(3, 5).astype(float)
        x_var = Parameter(x_np.copy())

        dp = Dropout(p=0.5)
        dp.training = True

        out_node = dp(x_var)
        out = out_node.value

        # Identify kept positions via comparing out to 0
        kept_mask = (out != 0)

        # where kept, value should be x * (1/(1-p))
        expected_kept_vals = x_np * (1.0 / (1.0 - dp.p))
        self.assertTrue(np.allclose(
            out[kept_mask], expected_kept_vals[kept_mask], atol=EPSILON))

        # In eval mode, dropout should be an identity function (out = in)
        dp.training = False
        x_var2 = Parameter(x_np.copy())
        out_eval = dp(x_var2).value
        self.assertTrue(np.allclose(out_eval, x_np, atol=EPSILON))

    def test_dropout_backward(self) -> None:
        x_np = np.random.randn(3, 5).astype(float)
        x_var = Parameter(x_np.copy())

        dp = Dropout(p=0.5)
        dp.training = True

        out_node = dp(x_var)
        out = out_node.value

        # Identify kept positions via comparing out to 0
        kept_mask = (out != 0)

        # gradient on x should be zero where mask is zero, scaled where kept
        out_node.backward(np.ones_like(out_node.value))
        x_grad = x_var.grad
        expected_grad = np.zeros_like(x_np)
        expected_grad[kept_mask] = 1.0 / (1.0 - dp.p)
        self.assertTrue(np.allclose(x_grad, expected_grad, atol=EPSILON))

        # In eval mode, dropout should be an identity function (out = in)
        dp.training = False
        x_var2 = Parameter(x_np.copy())
        out_eval_node = dp(x_var2)
        out_eval = out_eval_node.value
        self.assertTrue(np.allclose(out_eval, x_np, atol=EPSILON))

        # backward in eval is direct pass-through
        out_eval_node.backward(np.ones_like(out_eval_node.value))
        self.assertTrue(np.allclose(
            x_var2.grad, np.ones_like(x_np), atol=EPSILON))
