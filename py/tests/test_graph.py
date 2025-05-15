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
        pass

    def test_node_matrix_multiplication(self) -> None:
        a = Node(np.array([[0.17768567, 0.39771961, 0.92563573, 0.66784523],
                           [0.62486733, 0.40113929, 0.57849025, 0.61530686],
                           [0.55179607, 0.96602099, 0.57536875, 0.64842823]]))
        b = Node(np.array([[0.66110405, 0.86829835],
                           [0.66554035, 0.39163418],
                           [0.0042813, 0.71459745],
                           [0.23730456, 0.59034684]]))
        pass

    def test_node_negation(self) -> None:
        pass

    def test_top_sort(self) -> None:
        pass
