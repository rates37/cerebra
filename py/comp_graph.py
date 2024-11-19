from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from typing import List, Any, Union

##! Types:
ARR = Union[npt.NDArray, float]
GRAD = Union[None, npt.NDArray]


# Node class: abstract class for any node in a computational graph
class Node(object):
    def __init__(self) -> None:
        self.out: ARR
        self.out_grad: GRAD
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def get_predecessors(self) -> List[Node]:
        pass

    def zero_grad(self) -> None:
        # resets the gradient of the Node's output
        # pass
        if self.out:
            self.out_grad = np.zeros(self.out.shape)
        else:
            self.out_grad = np.zeros(1)


# Value Node class: a node for a constant value
class ValueNode(Node):
    def __init__(self, value: ARR) -> None:
        self.value = value
        self.out: ARR = self.value
        self.out_grad: GRAD = (
            None  # holds the gradient of self.out after backward pass is called
        )

    def forward(self) -> ARR:
        # reset gradient:
        self.zero_grad()
        
        # compute output:
        self.out = self.value

        # return output:
        return self.out

    def backward(self) -> ARR:
        return self.out_grad

    def get_predecessors(self) -> List[Node]:
        return []
    
    def set_value(self, value: ARR) -> None:
        self.value = value


##! Simple Operation Nodes:
class PlusNode(Node):
    # Node represents the operation x+y
    def __init__(self, x: Node, y: Node) -> None:
        self.x: Node = x
        self.y: Node = y
        self.out: ARR = None
        self.out_grad: GRAD = None

    def forward(self) -> ARR:
        # reset gradient:
        self.zero_grad()

        # compute output:
        self.out = self.x.out + self.y.out  #! recompute each time
        return self.out

    def backward(self) -> ARR:
        # compute partial derivatives of output w.r.t each input:
        x_grad_partial = self.out_grad  # d/dx (x+y) = 1
        y_grad_partial = self.out_grad  # d/dy (x+y) = 1

        # update gradients of predecessors
        self.x.out_grad += x_grad_partial
        self.y.out_grad += y_grad_partial

        return self.out_grad

    def get_predecessors(self) -> List[Node]:
        return [self.x, self.y]


class MultiplyNode(Node):
    # Node represents the operation x*y
    def __init__(self, x: Node, y: Node) -> None:
        self.x = x
        self.y = y
        self.out: ARR = None
        self.out_grad: GRAD = None

    def forward(self):
        # reset gradient:
        self.zero_grad()

        # compute output:
        self.out = self.x.out * self.y.out
        return self.out

    def backward(self):
        # assume that self.out_grad has been appropriately updated as a precondition

        # compute partial gradients of output w.r.t. each input:
        x_grad_partial = self.out_grad * self.y.out  # d/dx (x*y) = y
        y_grad_partial = self.out_grad * self.x.out  # d/dy (x*y) = x

        # update gradients of predecessors:
        self.x.out_grad += x_grad_partial
        self.y.out_grad += y_grad_partial

        # return own gradient:
        return self.out_grad

    def get_predecessors(self) -> List[Node]:
        return [self.x, self.y]




def testing() -> None:
    # create function h(x) = x + x^2 = plus(x, x^2) = plus(x, mult(x,x))
    x = ValueNode(None)
    mult = MultiplyNode(x,x)
    f = PlusNode(x, mult)
    
    values = [0,1,2,3]
    
    for v in values:
        print("\n===========")
        x.set_value(np.array(v))
        
        ## forward pass:
        x.forward()
        mult.forward()
        f.forward()
        # print outputs:
        print(f"\tx\t=\t{x.out}")
        print(f"\tx^2\t=\t{mult.out}")
        print(f"\tx^2+x\t=\t{f.out}")
        print(f"\tf(x)\t=\tx^2+x")
        
        ## backward pass:
        # set gradient of output w.r.t output equal to 1:
        f.out_grad = 1
        f.backward()
        mult.backward()
        x.backward()
        
        print(f"\tf'(x)\t=\t{x.out_grad.item()}")
        
    pass


if __name__ == "__main__":
    testing()