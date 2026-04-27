# Roadmap

This file outlines planned features, improvements, and ongoing goals.

## Planned

| Feature                             | Priority | Notes                                                                                   |
| ----------------------------------- | -------- | --------------------------------------------------------------------------------------- |
| More optimisers                     | Medium   | ADAM, Adagrad, etc.                                                                     |
| Flatten layer                       | Low      |                                                                                         |
| Graph pruning / memory optimisation | Medium   | E.g., ~~no_grad context~~, detach, etc                                                  |
| RNN layers                          | Medium   |                                                                                         |
| Optimise implementations            | Low      | Some features may be inefficiently implemented such as using for loops in Conv2d layers |
| Dataset and Dataloaders             | Medium   |                                                                                         |
| Trainer classes                     | Medium   |                                                                                         |
| Model saving / loading utility      | Medium   |                                                                                         |
| Transformer/Attention layers        | Medium   |                                                                                         |
| Learning Rate Schedulers            | Low      |                                                                                         |
| ConvTranspose layers                | Medium   |                                                                                         |
| Graph visualisation                 | Low      | Eg to mermaid diagrams                                                                  |

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
