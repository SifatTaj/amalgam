import torch
import torchvision
from tqdm import tqdm
import numpy as np

'''
Dataset Obfuscator

This module provides functionality to obfuscate datasets by augmenting samples with noise.
It supports loading datasets, generating noise, augmenting samples, and saving the augmented datasets.
'''

class DatasetObfuscator:
    def __init__(self, path: str, amount: float) -> None:
        self.path = path
        self.amount = amount
        self.n_samples = 0
        self.samples = None
        self.labels = None
        self.sample_shape = None
        self.aug_samples = None
        self.aug_shape = None
        self.aug_indices = None
        
        self.load()

    def load(self) -> None:
        try:
            data = torch.load(self.path)
            self.samples = data['images']
            self.labels = data['labels']

            self.n_samples = self.samples.shape[0]
            self.sample_shape = self.samples.shape[1:]

            self.aug_shape = (self.sample_shape[0], self.sample_shape[1] + int(self.sample_shape[1] * self.amount), self.sample_shape[2] + int(self.sample_shape[2] * self.amount))

            print(f"Dataset loaded successfully from {self.path}.")
            print(f"Number of samples: {self.samples.shape[0]}")
            print(f"Sample shape: {self.samples.shape}")

        except Exception as e:
            print(f"Failed to load dataset from {self.path}: {e}")

    def set_random_aug_indices(self, aug_indices: torch.Tensor) -> None:
        self.aug_indices = aug_indices

    def generate_random_indices(self, noise_shape: torch.Size) -> None:

        aug_indices = []
        aug_flat_size = self.aug_shape[1] * self.aug_shape[2]

        for c in range(self.sample_shape[0]):
            aug_indices.append(np.random.choice(np.arange(0, aug_flat_size), replace=False, size=noise_shape[2]))

        return torch.tensor(np.array(aug_indices))

    def get_random_aug_indices(self) -> torch.Tensor:
        if self.aug_indices is None:
            raise ValueError("Augmentation indices are not set. Call set_random_aug_indices() first.")

        return self.aug_indices

    def generate_noise(self, amount: float, type: str) -> None:
        dataset_size = self.samples.shape[0]
        noise_dim = (self.aug_shape[1] * self.aug_shape[2]) - (self.sample_shape[1] * self.sample_shape[2])
        noise_shape = (self.sample_shape[0], noise_dim)

        noise_set = []

        for i in range(dataset_size):
            if type == 'uniform':
                noise_set.append(torch.rand(noise_shape))

            elif type == 'gaussian' or type == 'normal':
                noise_set.append(torch.abs(torch.randn(noise_shape)).clamp(0, 1))

            else:
                raise ValueError("Unsupported noise type. Use 'uniform' or 'gaussian'/'normal'.")

        print(f"{len(noise_set)} noise generated of shape {noise_shape}.")
        return torch.stack(noise_set, dim=0)

    def augment_dataset(self, noise_set: torch.Tensor) -> torch.Tensor:
        if self.aug_indices is None:
            raise ValueError("Augmentation indices are not set. Set indices using set_random_aug_indices() first.")

        assert noise_set.shape[0] == self.samples.shape[0], "Noise set and samples must have the same number of samples."
        assert noise_set.shape[2] == (self.aug_shape[1] * self.aug_shape[2]) - (self.sample_shape[1] * self.sample_shape[2]), "Noise shape is incorrect."

        augmented_dataset = []

        print("Augmenting dataset...")
        for i in tqdm(range(self.samples.shape[0])):
            augmented_dataset.append(self._augment_sample(self.samples[i], noise_set[i], self.aug_indices))
        self.aug_samples = torch.stack(augmented_dataset, dim=0)

    def _augment_sample(self, sample_tensor: torch.Tensor, noise_tensor: torch.Tensor, aug_indices: list[int]) -> torch.Tensor:

        aug_sample = torch.zeros(self.aug_shape[0], self.aug_shape[1] * self.aug_shape[2])

        for c in range(sample_tensor.shape[0]):
            sample_flat = torch.flatten(sample_tensor[c])
            aug_flat = aug_sample[c]

            sample_iter = iter(sample_flat)
            noise_iter = iter(noise_tensor[c])

            for i in range(aug_sample.shape[1]):
                if i in aug_indices[c]:
                    aug_flat[i] = next(noise_iter)
                else:
                    aug_flat[i] = next(sample_iter)

        return aug_sample.reshape(self.aug_shape)

    def save_augmented_dataset(self, save_path: str) -> None:
        if self.aug_samples is None:
            raise ValueError("Augmented samples are not available. Call augment_dataset() first.")

        data_to_save = {
            'images': self.aug_samples,
            'labels': self.labels,
            'aug_indices': self.aug_indices
        }

        try:
            torch.save(data_to_save, save_path)
            print(f"Augmented dataset saved successfully to {save_path}.")
        except Exception as e:
            print(f"Failed to save augmented dataset to {save_path}: {e}")