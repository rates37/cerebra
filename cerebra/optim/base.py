from ..nn.module import Parameter
from typing import List
import numpy as np
from abc import ABC, abstractmethod

class Optimiser(ABC):
    """Abstract Optimiser base class.
    """
    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        """Initialises the optimiser.   

        Args:
            parameters (List[Parameter]): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Performs a single optimisation step.
        """
        pass

    def zero_grad(self) -> None:
        """Clears the gradients of all optimised parameters.
        
        This should be called before computing the gradients for the next 
        optimisation step (typically before `loss.backward()`).
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.value)
