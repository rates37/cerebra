import unittest
import numpy as np

from cerebra import Node

EPSILON = 1e-6


class TestNode(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rng = np.random.default_rng(69)

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

    def test_node_multiplication(self) -> None:
        # Test 1: Scalar * Scalar
        a = Node(4)
        b = Node(5)
        c = a * b
        self.assertTrue(np.allclose(c.value, np.array([20]), atol=EPSILON))

        # Test 2: Simple vector dot multiplication
        a = Node(np.array([1, 2, 3]))
        b = Node(np.array([4, 5, 6]))
        c = a * b
        self.assertTrue(np.allclose(c.value, np.array([4, 10, 18]), atol=EPSILON))

        # Test 3: Matrix dot multiplication
        a = Node(np.array([[1, 2, 3], [4, 5, 6]]))
        b = Node(np.array([[4, 5, 6], [1, 2, 3]]))
        c = a * b
        self.assertTrue(np.allclose(c.value, np.array(
            [[4, 10, 18], [4, 10, 18]]), atol=EPSILON))

        # Test 4: Random values
        a = Node(np.array([[0.66110405, 0.86829835],
                           [0.66554035, 0.39163418],
                           [0.0042813, 0.71459745],
                           [0.23730456, 0.59034684]]))
        b = Node(np.array([[0.13673004, 0.85853164],
                           [0.7216384, 0.92465808],
                           [0.32085063, 0.4999047],
                           [0.38680514, 0.14657554]]))
        c = a * b
        self.assertTrue(np.allclose(c.value, np.array([[0.09039278, 0.74546161], 
                                                       [0.48027947, 0.36212771], 
                                                       [0.00137366, 0.35723062], 
                                                       [0.09179062, 0.08653041]]), atol=EPSILON))
        
        # Test 5: Vector * Scalar (broadcasting)
        a = Node(np.array([1.0, 2.0]))
        b_val = 3.0
        c = a * b_val
        self.assertTrue(np.allclose(c.value, np.array([3.0, 6.0]), atol=EPSILON))

        # Test 6: Scalar * Vector (broadcasting)
        a_val = 4.0
        b = Node(np.array([3, 6]))
        c = a_val * b
        self.assertTrue(np.allclose(c.value, np.array([12.0, 24.0]), atol=EPSILON))


    def test_node_subtraction(self) -> None:
        # Test 1: Vector - vector
        a = Node(np.array([10,20,30]))
        b = Node(np.array([1,2,3]))
        c = a - b
        self.assertTrue(np.allclose(c.value, np.array([9,18,27]), atol=EPSILON))

        # Test 2: Vector - scalar
        a = Node(np.array([10.0, 20.0]))
        b_val = 3.0
        c = a - b_val
        self.assertTrue(np.allclose(c.value, np.array([7.0, 17.0]), atol=EPSILON))

        # Test 3: Scalar - vector
        a_val = 100.0
        b = Node(np.array([10, 20]))
        c = a_val - b
        self.assertTrue(np.allclose(c.value, np.array([90, 80]), atol=EPSILON))

        # Test 4: Matrix - matrix:
        a = Node(np.array([1, 2, 3]))
        b = Node(np.array([4, 5, 6]))
        c = a - b
        self.assertTrue(np.allclose(c.value, np.array([-3, -3, -3]), atol=EPSILON))


    def test_node_matrix_multiplication(self) -> None:
        # Test 1: Node @ Node
        a_val = self.default_rng.random((2,3))
        b_val = self.default_rng.random((3,4))
        a = Node(a_val)
        b = Node(b_val)
        c = a @ b
        self.assertTrue(np.allclose(c.value, a_val @ b_val, atol=EPSILON))

        # Test 2: Node @ np.ndarray
        d = a @ b_val
        self.assertTrue(np.allclose(d.value, a_val @ b_val, atol=EPSILON))

        #! NOTE: this doesn't work, since Python tries to call the __mul__ method
        #!  of the LEFT operand first
        # # Test 3: np.ndarray @ Node
        # e = a_val @ b
        # self.assertTrue(np.allclose(e.value, a_val @ b_val, atol=EPSILON))


    def test_node_negation(self) -> None:
        a = Node(np.array([1,-2,0]))
        b = -a
        self.assertTrue(np.allclose(b.value, np.array([-1,2,0]), atol=EPSILON))
        self.assertEqual(len(b.parents), 1)

    def test_top_sort(self) -> None:
        pass
