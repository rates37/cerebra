# Roadmap

This file outlines planned features, improvements, and ongoing goals.

## Planned

| Feature                             | Priority | Notes                                                                                   |
| ----------------------------------- | -------- | --------------------------------------------------------------------------------------- |
| Graph pruning / memory optimisation | Medium   | E.g., ~~no_grad context~~, detach, etc                                                  |
| Transformer/Attention layers        | Medium   |                                                                                         |
| Model saving / loading utility      | Medium   |                                                                                         |
| Regularisation                      | Medium   |                                                                                         |
| Graph visualisation                 | Low      | Eg to mermaid diagrams                                                                  |
| Dataset and Dataloaders             | Medium   |                                                                                         |
| Trainer classes                     | Medium   |                                                                                         |
| Optimise implementations            | Low      | Some features may be inefficiently implemented such as using for loops in Conv2d layers |
| More optimisers                     | Medium   | ADAM, Adagrad, etc.                                                                     |
| Learning Rate Schedulers            | Low      |                                                                                         |
| RNN layers                          | Medium   |                                                                                         |
| ConvTranspose layers                | Medium   |                                                                                         |
| Flatten layer                       | Low      |                                                                                         |

---

## Completed Features

| Feature              | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `no_grad`            | A context manager that disables automatic gradient tracking               |
| Activation functions | Added common activations functions `sigmoid`, `tanh`, `elu`, `leaky_relu` |
| Dropout              | Added dropout layers, works with any generalised input shape              |
| Unit Test Coverage   | Added unit tests for all core components                                  |
| Normalisation        | Added batchnorm and layernorm layers                                      |
| Documentation        | Added docstrings                                                          |

---

## Community Suggestions

Suggestions and feature requests are welcomed, please make an issue (if it doesn't already exist) with the enhancement tag.

---
