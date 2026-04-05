from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
import numpy as np
from ..core.node import Variable

class Parameter(Variable):
    """A variable considered as a trainable parameter of a module."""
    def __init__(self, value: Union[np.ndarray, float, int], name: Optional[str] = None) -> None:
        """Initialises a parameter with a value and an optional name.

        Args:
            value (Union[np.ndarray, float, int]): The initial value for the parameter.
            name (Optional[str], optional): The name of the parameter. Defaults to None.
        """
        super().__init__(value, name=name)

class Module(ABC):
    """Base class for all neural network modules.
    
    Models should also subclass this class. Models can contain other Modules, 
    allowing to nest them in a tree structure. Very similar to Pytorch modules.
    """
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Defines the computation performed at every call.
        
        Should be overridden by all subclasses.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The output of the forward computation.
        """
        pass

    def parameters(self) -> List[Parameter]:
        """Returns a list of all parameters in the module and its submodules.

        Returns:
            List[Parameter]: A list containing all the `Parameter` instances 
                registered in this module and its descendants.
        """
        params = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                params.append(attribute)
            elif isinstance(attribute, Module):
                params.extend(attribute.parameters())
        return params
