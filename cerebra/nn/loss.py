import numpy as np
from typing import Union, List
from ..core.node import Node, to_node
from ..core.ops import Operation

class CrossEntropyLoss(Operation):
    """Cross Entropy Loss operation.
    
    This criterion computes the cross entropy loss between input logits and target.
    """
    def __init__(self, target: np.ndarray) -> None:
        """Initialises the cross entropy loss with target indices.
        
        Args:
            target (np.ndarray): Target class indices, expected to be shape (N,).
        """
        self.target = target

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the cross entropy loss.
        
        Args:
            x (np.ndarray): The input logits with shape (batch_size, num_classes).
            
        Returns:
            np.ndarray: The computed cross entropy loss as a scalar array.
        """
        # presume x in format: (batch_size, num_classes)
        x_cpy = x.copy()
        x_max = np.max(x_cpy, axis=1, keepdims=True)
        x_exp = np.exp(x_cpy - x_max)
        sum_exp = np.sum(x_exp, axis=1, keepdims=True)
        self.softmax = x_exp / sum_exp
        batch_size = x_cpy.shape[0]
        log_probabilities = np.log(self.softmax[np.arange(
            batch_size), self.target] + 1e-12)  # prevent instabilities
        loss_val = -np.mean(log_probabilities)
        return np.array([loss_val]) # convert to scalar np array to follow Node method type hint

    def backward(self, output_grad: np.ndarray, node: Node) -> List[np.ndarray]:
        batch_size, _ = self.softmax.shape
        one_hot_logits = np.zeros_like(self.softmax)
        one_hot_logits[np.arange(batch_size), self.target] = 1.0
        grad = (self.softmax - one_hot_logits) / batch_size
        grad = grad * output_grad
        return [grad]


# function for CELoss so can be used as functional block:
def cross_entropy_loss(x: Union[Node, np.ndarray, float, int], target: np.ndarray) -> Node:
    """Computes the cross entropy loss between input logits and target indices.

    Args:
        x (Union[Node, np.ndarray, float, int]): The predicted logits.
        target (np.ndarray): The ground truth target indices.

    Returns:
        Node: A Node containing the calculated loss value.
    """
    x = to_node(x)
    op = CrossEntropyLoss(target)
    val = op.forward(x.value)
    return Node(val, parents=[x], op=op)
