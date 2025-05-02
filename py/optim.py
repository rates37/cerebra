from network import Parameter
from typing import List
import numpy as np


class SGD:
    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr
    
    def step(self) -> None:
        for p in self.parameters:
            p.value = p.value - self.lr * p.grad
    
    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = np.zeros_like(p.value)
