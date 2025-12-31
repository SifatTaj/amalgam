# Amalgam: A Framework for Obfuscated Neural Network Training on the Cloud

![diagram](assets/amalgam.png)

Amalgam is an easy-to-use neural network obfuscation framework. Users simply need to upload a PyTorch model and a dataset. 
Currently, Amalgam supports vision language models written using the PyTorch framework.

Amalgam augments PyTorch models and datasets to be used for training with well-calibrated noise to “hide” both the original
model architectures and training datasets from the cloud.
After training, Amalgam extracts the original models from the augmented models and returns them to users.

## Features

- A simple-to-use API for PyTorch models and datasets
- Currently works with computer vision models and datasets
- Supports CUDA (Nvidia) and MPS (Apple Silicon) acceleration
- Supports fine-tuning an existing PyTorch model
- Incurs a moderate overhead compared to other privacy-preserving frameworks
- No accuracy loss

## Use Case

Consider a scenario where you have a CIFAR-10 dataset that you want to train on a cloud provider, but you need to protect the original dataset from exposure. Amalgam simplifies this process:

1. **Initialize the obfuscator** with your original dataset and specify the amount of augmentation (e.g., 25% of samples):
   ```python
   obfuscator = DatasetObfuscator('datasets/cifar10_train.pt', amount=0.25)
   ```

2. **Generate calibrated noise** using a uniform distribution to augment the dataset:
   ```python
   noise = obfuscator.generate_noise(0.25, 'uniform')
   aug_indices = obfuscator.generate_random_indices(noise.shape)
   ```

3. **Apply obfuscation** by augmenting the dataset with the generated noise:
   ```python
   obfuscator.set_random_aug_indices(aug_indices)
   obfuscator.augment_dataset(noise)
   ```

4. **Save the obfuscated dataset** for secure cloud training:
   ```python
   obfuscator.save_augmented_dataset('datasets/cifar10_train_obfuscated.pt')
   ```

The obfuscated dataset can now be uploaded to the cloud for model training without exposing your original data. After training completes, Amalgam can recover the original model, maintaining both privacy and model accuracy.

