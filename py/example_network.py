import numpy as np
from cerebra.network import cross_entropy_loss, Linear, Module, Variable
from cerebra.graph import Node, relu
from cerebra.optim import SGD
from typing import Tuple


class MLP(Module):
    """
    A simple MLP with a single hidden layer and ReLU activation on hidden layer.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features)
        self.fc2 = Linear(hidden_features, out_features)

    def forward(self, x: Node) -> Node:
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x


# function to download MNIST and downsample it:
def load_and_downsample_mnist(num_samples: int = 1000, downsample_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads MNIST using keras and downsamples each image by the provided factor.
    Images are flattened and normalised
    """
    # download data:
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # combine train and test sets:
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # shuffle data:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices][:num_samples]
    y = y[indices][:num_samples]

    # downsample:
    downsampled = []
    for img in X:
        img_downsampled = img[::downsample_factor, ::downsample_factor]
        downsampled.append(img_downsampled.flatten())
    X_down = np.stack(downsampled).astype(np.float32) / 255.0
    return X_down, y


if __name__ == "__main__":
    num_samples = 2000
    batch_size = 100
    num_epochs = 50
    learning_rate = 0.075
    downsample_factor = 2

    input_dimension = (28 // downsample_factor) ** 2
    hidden_dimension = 64
    num_classes = 10

    # load mnist:
    X, y = load_and_downsample_mnist(
        num_samples=num_samples, downsample_factor=downsample_factor)

    # instatiate model and optimiser:
    model = MLP(in_features=input_dimension,
                hidden_features=hidden_dimension, out_features=num_classes)
    optim = SGD(model.parameters(), lr=learning_rate)

    num_batches = X.shape[0] // batch_size

    for epoch in range(num_epochs):
        # shuffle data:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for i in range(num_batches):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            # wrap input in a variable
            X_input = Variable(X_batch)
            logits = model(X_input)
            loss = cross_entropy_loss(logits, y_batch)
            epoch_loss += loss.value.item()

            # backward pass:
            loss.backward()

            # run optimiser:
            optim.step()
            optim.zero_grad()

            # compute accuracy:
            exp_logits = np.exp(
                logits.value - np.max(logits.value, axis=1, keepdims=True))
            probabilities = exp_logits / \
                (np.sum(exp_logits, axis=1, keepdims=True))
            preds = np.argmax(probabilities, axis=1)
            correct_preds += np.sum(preds == y_batch)
            total_preds += y_batch.shape[0]
        loss_avg = epoch_loss / num_batches
        accuracy = (correct_preds / total_preds) * 100
        print(
            f"Epoch: {epoch}: Loss: {loss_avg:.4f}, Accuracy = {accuracy:.2f}%")
