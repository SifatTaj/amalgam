# Amalgam: A Framework for Obfuscated Neural Network Training on the Cloud

Amalgam is an easy-to-use neural network obfuscation framework. Users simply need to upload a PyTorch model and a dataset. 
Currently, Amalgam supports vision language models written using the PyTorch framework.

Amalgam augments PyTorch models and datasets to be used for training with well-calibrated noise to “hide” both the original
model architectures and training datasets from the cloud.
After training, Amalgam extracts the original models from the augmented models and returns them to users.

## Features

---

- A simple-to-use API for PyTorch models and datasets
- Currently works with computer vision models and datasets
- Supports CUDA (Nvidia) and MPS (Apple Silicon) acceleration
- Supports fine-tuning an existing PyTorch model
- Incurs a moderate overhead compared to other privacy-preserving frameworks
- No accuracy loss
```
