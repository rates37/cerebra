from ..nn.module import Parameter
from typing import List
import numpy as np

class SGD:
    """Stochastic Gradient Descent optimiser.
    """
    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        """Initialises the SGD optimiser.   

        Args:
            parameters (List[Parameter]): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Performs a single optimisation step.
        
        Updates the parameters using their computed gradients according to 
        the SGD update rule: $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$
        """
        for p in self.parameters:
            p.value = p.value - self.lr * p.grad

    def zero_grad(self) -> None:
        """Clears the gradients of all optimised parameters.
        
        This should be called before computing the gradients for the next 
        optimisation step (typically before `loss.backward()`).
        """
        for p in self.parameters:
            p.grad = np.zeros_like(p.value)
