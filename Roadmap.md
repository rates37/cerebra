# Roadmap

This file outlines planned features, improvements, and ongoing goals.

## Planned

| Feature                             | Priority | Notes                                                                                   |
| ----------------------------------- | -------- | --------------------------------------------------------------------------------------- |
| Documentation                       | High     | Add docstrings and examples                                                             |
| Batchnorm                           | Medium   |                                                                                         |
| Layernorm                           | Medium   |                                                                                         |
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

| Feature              | Description                                                               | Released In |
| -------------------- | ------------------------------------------------------------------------- | ----------- |
| `no_grad`            | A context manager that disables automatic gradient tracking               | v0.0.1      |
| Activation functions | Added common activations functions `sigmoid`, `tanh`, `elu`, `leaky_relu` | v0.0.1      |
| Dropout              | Added dropout layers, works with any generalised input shape              | v0.0.1      |
| Unit Test Coverage   | Added unit tests for all core components                                  | v0.0.1      |

---

## Community Suggestions

Suggestions and feature requests are welcomed, please make an issue (if it doesn't already exist) with the enhancement tag.

---
