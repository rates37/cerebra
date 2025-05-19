import unittest
import numpy as np

from cerebra import Node, Conv2d, Parameter, Operation

EPSILON = 1e-6


class TestConv2d(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

    def tearDown(self) -> None:
        pass

    #! =========================
    #!    Forward Conv Tests
    #! =========================

    def test_conv2d_forward_no_bias(self):
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        # expected value is: (1*1 + 2*0 + 3*0 + 4*1) = 5
        expected_val = np.array([5])
        op = Conv2d(stride=1, padding=0)
        output = op.forward(x=x_val, weight=w_val, bias=None)

        self.assertTrue(np.allclose(output, expected_val, atol=EPSILON))

    def test_conv2d_forward_with_bias(self):
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        b_val = np.array([0.5], dtype=np.float32)
        # expected value is: (1*1 + 2*0 + 3*0 + 4*1) + 0.5 = 5.5
        expected_val = np.array([5.5])
        op = Conv2d(stride=1, padding=0)
        output = op.forward(x=x_val, weight=w_val, bias=b_val)

        self.assertTrue(np.allclose(output, expected_val, atol=EPSILON))

    def test_conv2d_op_forward_padding_stride(self):
        op = Conv2d(stride=2, padding=1)
        x_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        w_val = np.array([[[[1, 1], [1, 1]]]], dtype=np.float32)
        # Padded_x:
        # 0 0 0 0
        # 0 1 2 0
        # 0 3 4 0
        # 0 0 0 0

        # output dimensions:
        # out_h = (2+2*1-2)//2 + 1 = 2
        # out_w = (2+2*1-2)//2 + 1 = 2

        # output matrix entries:
        # Top left from padded x[0:2, 0:2] = [[0,0],[0,1]] -> sum = 1
        # Top right from padded x[0:2, 2:4] = [[0,0],[2,0]] -> sum = 2
        # Bottom left from padded x[2:4, 0:2] = [[0,3],[0,0]] -> sum = 3
        # Bottom right from padded x[2:4, 2:4] = [[4,0],[0,0]] -> sum = 4
        expected_out = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        out = op.forward(x_val, w_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_conv2d_op_forward_multi_channel_in(self):
        op = Conv2d(stride=1, padding=0)
        # C_in=2, H_in=1, W_in=1
        x_val = np.array([[[[10.]], [[20.]]]], dtype=np.float32)
        w_val = np.array([[[[2.]], [[3.]]]], dtype=np.float32)
        b_val = np.array([0.5], dtype=np.float32)
        # Expected: (10*2 + 20*3) + 0.5 = 80.5
        expected_out = np.array([[[[80.5]]]], dtype=np.float32)
        out = op.forward(x_val, w_val, b_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    def test_conv2d_op_forward_multi_channel_out(self):
        op = Conv2d(stride=1, padding=0)
        # C_out=2
        x_val = np.array([[[[5]]]], dtype=np.float32)
        w_val = np.array([[[[2]]], [[[3]]]], dtype=np.float32)
        b_val = np.array([0.1, 0.2], dtype=np.float32)

        # Expected:
        # Channel 0: (5*2) + 0.1 = 10.1
        # Channel 1: (5*3) + 0.2 = 15.2
        expected_out = np.array([[[[10.1]], [[15.2]]]], dtype=np.float32)
        out = op.forward(x_val, w_val, b_val)
        self.assertTrue(np.allclose(out, expected_out, atol=EPSILON))

    #! ==========================
    #!    Backward Conv Tests
    #! ==========================
