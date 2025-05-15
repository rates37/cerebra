import unittest
import numpy as np

from cerebra import Node

EPSILON = 1e-6


class TestNode(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_node_addition(self) -> None:
        # Test 1: Node + Node
        a = Node(np.array([1, 2, 3]))
        b = Node(np.array([4, 5, 6]))

        c = a+b
        self.assertTrue(np.allclose(
            c.value, np.array([5, 7, 9]), atol=EPSILON))

        # check parents:
        self.assertEqual(len(c.parents), 2)
        self.assertIs(c.parents[0], a)
        self.assertIs(c.parents[1], b)

        # Test 2: Node + Scalar value (broadcasting)
        a = Node(np.array([1.0, 2.0, 3.0]))
        b_val = 5
        c = a + b_val
        self.assertTrue(np.allclose(
            c.value, np.array([6.0, 7.0, 8.0]), atol=EPSILON))
        self.assertEqual(len(c.parents), 2)
        self.assertIs(c.parents[0], a)
        self.assertTrue(np.allclose(c.parents[1].value, np.array(b_val)))

        # Test 3: Scalar + Node (broadcasting)
        a_val = 10.0
        b = Node(np.array([4, 5, 6]))
        c = a_val + b
        self.assertTrue(np.allclose(
            c.value, np.array([14, 15, 16]), atol=EPSILON))
        self.assertEqual(len(c.parents), 2)
        self.assertTrue(np.allclose(c.parents[0].value, np.array(a_val)))
        self.assertIs(c.parents[1], b)

        # Test 4: Node + numpy array:
        a = Node(np.array([[1, 2], [3, 4]]))
        b_array = np.array([[10, 20], [30, 40]])
        c = a + b_array
        self.assertTrue(np.allclose(c.value, np.array(
            [[11, 22], [33, 44]]), atol=EPSILON))
        self.assertEqual(len(c.parents), 2)
        self.assertIs(c.parents[0], a)
        self.assertTrue(np.allclose(c.parents[1].value, b_array))
