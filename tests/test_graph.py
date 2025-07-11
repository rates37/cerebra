import unittest
import numpy as np

from cerebra import Node, relu

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
        self.assertTrue(np.allclose(
            c.value, np.array([4, 10, 18]), atol=EPSILON))

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
        self.assertTrue(np.allclose(
            c.value, np.array([3.0, 6.0]), atol=EPSILON))

        # Test 6: Scalar * Vector (broadcasting)
        a_val = 4.0
        b = Node(np.array([3, 6]))
        c = a_val * b
        self.assertTrue(np.allclose(
            c.value, np.array([12.0, 24.0]), atol=EPSILON))

    def test_node_subtraction(self) -> None:
        # Test 1: Vector - vector
        a = Node(np.array([10, 20, 30]))
        b = Node(np.array([1, 2, 3]))
        c = a - b
        self.assertTrue(np.allclose(
            c.value, np.array([9, 18, 27]), atol=EPSILON))

        # Test 2: Vector - scalar
        a = Node(np.array([10.0, 20.0]))
        b_val = 3.0
        c = a - b_val
        self.assertTrue(np.allclose(
            c.value, np.array([7.0, 17.0]), atol=EPSILON))

        # Test 3: Scalar - vector
        a_val = 100.0
        b = Node(np.array([10, 20]))
        c = a_val - b
        self.assertTrue(np.allclose(c.value, np.array([90, 80]), atol=EPSILON))

        # Test 4: Matrix - matrix:
        a = Node(np.array([1, 2, 3]))
        b = Node(np.array([4, 5, 6]))
        c = a - b
        self.assertTrue(np.allclose(
            c.value, np.array([-3, -3, -3]), atol=EPSILON))

    def test_node_matrix_multiplication(self) -> None:
        # Test 1: Node @ Node
        a_val = self.default_rng.random((2, 3))
        b_val = self.default_rng.random((3, 4))
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
        a = Node(np.array([1, -2, 0]))
        b = -a
        self.assertTrue(np.allclose(
            b.value, np.array([-1, 2, 0]), atol=EPSILON))
        self.assertEqual(len(b.parents), 1)

    def test_top_sort(self) -> None:
        # Test 1: Complex layered graph
        a = Node(1, name="a")
        b = Node(2, name="b")
        c = a + b
        c.name = "c"  # c = a+b
        d = c * a
        d.name = "d"  # d = c*a
        e = relu(d)
        e.name = "e"  # e = relu(d) = relu(c*a) = relu((a+b)a)

        order = e.top_sort_ancestors()

        # Expected orders: (a, b, c, d, e) or (b, a, c, d, e)
        # The key is that parents must come before children
        self.assertEqual(len(order), 5)
        self.assertIn(a, order)
        self.assertIn(b, order)
        self.assertIn(c, order)
        self.assertIn(d, order)
        self.assertIn(e, order)

        # need to explicitly check relative ordering since multiple valid topological ordering
        #  may exist
        self.assertLess(order.index(a), order.index(c))
        self.assertLess(order.index(b), order.index(c))
        self.assertLess(order.index(c), order.index(d))
        self.assertLess(order.index(a), order.index(d))
        self.assertLess(order.index(d), order.index(e))
        self.assertIs(order[-1], e)  # The node itself should be last

        # Test 2: Test with a single node
        f = Node(10, name="f")
        order_f = f.top_sort_ancestors()
        self.assertEqual(len(order_f), 1)
        self.assertIs(order_f[0], f)

    def test_backward_simple_add(self) -> None:
        # Test 1: vector + vector
        a = Node(np.array([1, 2]))
        b = Node(np.array([3, 4]))
        c = a + b
        c.backward()

        # check gradients exist:
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(c.grad)

        # check gradients are correct:
        self.assertTrue(np.allclose(c.grad, np.array([1, 1]), atol=EPSILON))
        self.assertTrue(np.allclose(a.grad, np.array([1, 1]), atol=EPSILON))
        self.assertTrue(np.allclose(b.grad, np.array([1, 1]), atol=EPSILON))

        # todo: add more tests

    def test_backward_simple_mul(self) -> None:
        # Test 1: vector + vector
        a = Node(np.array([1, 2]))
        b = Node(np.array([3, 4]))
        c = a * b
        c.backward()

        # check gradients exist:
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertIsNotNone(c.grad)

        # check gradients are correct:
        self.assertTrue(np.allclose(c.grad, np.array(
            [1, 1]), atol=EPSILON))  # dL/dc = 1
        # dL/da = dL/dc * dc/da = 1 * d/da(a*b) = b
        self.assertTrue(np.allclose(a.grad, b.value, atol=EPSILON))
        # dL/db = dL/dc * dc/db = 1 * d/db(a*b) = a
        self.assertTrue(np.allclose(b.grad, a.value, atol=EPSILON))

        # todo: add more tests

    def test_backward_chain_rule(self) -> None:
        x = Node(2.0, name="x")
        y = Node(3.0, name="y")
        z = Node(4.0, name="z")

        # f = (x*y) + z
        a = x * y  # a = x*y = 6
        a.name = "a"
        f = a + z  # f = a+z = 10
        f.name = "f"

        f.backward()  # dL/df = 1

        # dL/df = 1
        # dL/da = dL/df * df/da = 1 * 1 = 1
        # dL/dz = dL/df * df/dz = 1 * 1 = 1
        # dL/dx = dL/da * da/dx = 1 * y = 3
        # dL/dy = dL/da * da/dy = 1 * x = 2

        self.assertTrue(np.allclose(f.grad, np.array(1.0)))
        self.assertTrue(np.allclose(a.grad, np.array(1.0)))
        self.assertTrue(np.allclose(z.grad, np.array(1.0)))
        self.assertTrue(np.allclose(y.grad, np.array(x.value)))
        self.assertTrue(np.allclose(x.grad, np.array(y.value)))

    def test_backward_broadcasting(self) -> None:
        x_val = np.array([[1., 2.], [3., 4.]])
        y_val = np.array([[10., 20.]])  # Will broadcast to [[10,20],[10,20]]
        x = Node(x_val)
        y = Node(y_val)

        # Test addition
        z_addition = x+y
        z_addition.backward()

        # dL/dx = dL / dz * dz/dx = [[1,1], [1,1]]
        self.assertTrue(np.allclose(
            x.grad, np.array([[1, 1], [1, 1]]), atol=EPSILON))
        # dL/dy = dL / dz * dz/dy, summed along axis 0 = [[2,2], [2,2]]
        self.assertTrue(np.allclose(y.grad, np.array([[2, 2]]), atol=EPSILON))

        # Reset values for next test:
        x_val = np.array([[1., 2.], [3., 4.]])
        y_val = np.array([[10., 20.]])  # Will broadcast to [[10,20],[10,20]]
        x = Node(x_val)
        y = Node(y_val)

        # Test multiplication
        z_multiplication = x*y
        z_multiplication.backward()

        # dL / dx = dL / dz * dz/dx = dL/dz * y broadcasted
        self.assertTrue(np.allclose(
            x.grad, np.array([y_val, y_val]), atol=EPSILON))

        # dL / dy = dL / dz * dz/dy = dL/dz * x summed along axis 0
        self.assertTrue(np.allclose(
            y.grad, np.sum(x_val, axis=0), atol=EPSILON))
