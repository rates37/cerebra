import unittest
import numpy as np
from cerebra import Parameter, Module
from tests.utils import EPSILON


class TestModule(unittest.TestCase):
    def test_parameter_collection(self) -> None:
        # Simple module
        class SimpleModule(Module):
            def __init__(self):
                self.p1 = Parameter(np.array([1.0]))
                self.p2 = Parameter(np.array([2.0]))

            def forward(self, x):
                return x

        model = SimpleModule()
        params = model.parameters()
        self.assertEqual(len(params), 2)
        self.assertIn(model.p1, params)
        self.assertIn(model.p2, params)

    def test_nested_module_parameters(self) -> None:
        # Nested modules
        class SubModule(Module):
            def __init__(self):
                self.p3 = Parameter(np.array([3.0]))

            def forward(self, x):
                return x

        class ParentModule(Module):
            def __init__(self):
                self.sub = SubModule()
                self.p4 = Parameter(np.array([4.0]))

            def forward(self, x):
                return self.sub(x)

        model = ParentModule()
        params = model.parameters()
        self.assertEqual(len(params), 2)
        self.assertIn(model.p4, params)
        self.assertIn(model.sub.p3, params)
