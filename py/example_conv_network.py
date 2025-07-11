import numpy as np
from cerebra.network import cross_entropy_loss, Linear, Module, Variable, Conv2dLayer
from cerebra.graph import Node, relu, reshape
from cerebra.optim import SGD
from typing import Tuple

class CNN(Module):
    def __init__(self, image_size: int=14) -> None:
        super().__init__()
        # create layers:
        self.c1 = Conv2dLayer(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
        self.c2 = Conv2dLayer(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc = Linear(16 * image_size * image_size, 10)
    
    def forward(self, x: Node) -> Node:
        # reshape input tensors from 196x1 to 14x14 tensors:
        N = x.value.shape[0]
        x = reshape(x, (N, 1, 14, 14))
        x = relu(self.c1(x))
        x = relu(self.c2(x))
        N,C,H,W = x.value.shape
        # flatten conv2d feature maps:
        x = reshape(x, (N, C*H*W))
        # apply classifier / dense layer:
        return self.fc(x)


# function to download MNIST and downsample it:
def load_and_downsample_mnist(num_samples: int = 1000, downsample_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads MNIST and downsamples each image by the provided factor.
    Images are flattened and normalised.
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1)
    
    # convert to np array
    X = mnist.data.to_numpy().reshape(-1, 28, 28)
    y = mnist.target.to_numpy().astype(np.int32)

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


    # load mnist:
    X, y = load_and_downsample_mnist(
        num_samples=num_samples, downsample_factor=downsample_factor)

    # instatiate model and optimiser:
    model = CNN(image_size=(28 // downsample_factor))
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
