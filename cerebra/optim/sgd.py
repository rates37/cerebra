from ..nn.module import Parameter
from .base import Optimiser
from typing import List
import numpy as np

class SGD(Optimiser):
    """Stochastic Gradient Descent optimiser.
    """
    def step(self) -> None:
        """Performs a single optimisation step.
        
        Updates the parameters using their computed gradients according to 
        the SGD update rule: $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$
        """
        for p in self.parameters:
            p.value = p.value - self.lr * p.grad

