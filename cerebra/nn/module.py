from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
import numpy as np
from ..core.node import Variable

class Parameter(Variable):
    def __init__(self, value: Union[np.ndarray, float, int], name: Optional[str] = None) -> None:
        super().__init__(value, name=name)

class Module(ABC):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def parameters(self) -> List[Parameter]:
        params = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                params.append(attribute)
            elif isinstance(attribute, Module):
                params.extend(attribute.parameters())
        return params
