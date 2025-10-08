# Roadmap

This file outlines planned features, improvements, and ongoing goals.

## Planned

| Feature | Priority | Notes |
|--------|----------|-------|
| Unit Test Coverage | High |  |
| Documentation | High | Add docstrings and examples |
| Dropout | Medium |  |
| Batchnorm | Medium |  |
| Layernorm | Medium |  |
| Graph pruning / memory optimisation | Medium | E.g., ~~no_grad context~~, detach, etc |
| Transformer/Attention layers | Medium |  |
| Model saving / loading utilty | Medium |  |
| Regularisation | Medium |  |
| Graph visualisation | Low | Eg to mermaid diagrams |
| Dataset and Dataloaders | Medium |  |
| Trainer classes | Medium |  |
| More activation functions | Medium | E.g., tanh, sigmoid, etc. |
| Optimise implementations | Low | Some features may be inefficiently implemented such as using for loops in Conv2d layers |
| More optimisers | Medium | ADAM, Adagrad, etc. |
| Learning Rate Schedulers | Low | |
| RNN layers | Medium | |
| ConvTranspose layers | Medium | |
| Flatten layer | Low | |

---

## Completed Features (todo)

| Feature | Description | Released In |
|--------|-------------|-------------|
| `no_grad` | A context manager that disables automatic gradient tracking | v0.0.1 |

---

## Community Suggestions

Suggestions and feature requests are welcomed, please make an issue (if it doesn't already exist) with the enhancement tag.

---
