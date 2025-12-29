import torch
import torchvision
import argparse
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class DatasetObfuscator:
    def __init__(self, path: str) -> None:
        self.path = path
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

            print(f"Dataset loaded successfully from {self.path}.")
            print(f"Number of samples: {self.samples.shape[0]}")
            print(f"Sample shape: {self.samples.shape}")

        except Exception as e:
            print(f"Failed to load dataset from {self.path}: {e}")

    def set_random_aug_indices(self, noise_shape: torch.Size) -> None:
        
        noise_size = noise_shape[1] * noise_shape[2]

        aug_indices = []
        aug_flat_size = (self.sample_shape[1] + noise_shape[1]) * (self.sample_shape[2] + noise_shape[2])
        for c in range(self.sample_shape[0]):
            aug_indices.append(np.random.choice(np.arange(0, aug_flat_size), replace=False, size=noise_size))

        self.aug_indices = torch.tensor(np.array(aug_indices))

    def get_random_aug_indices(self) -> torch.Tensor:
        if self.aug_indices is None:
            raise ValueError("Augmentation indices are not set. Call set_random_aug_indices() first.")

        return self.aug_indices

    def generate_noise(self, amount: float, type: str) -> None:
        dataset_size = self.samples.shape[0]
        noise_shape = (self.sample_shape[0], int(self.sample_shape[1] * amount), int(self.sample_shape[2] * amount))

        noise_set = []

        for i in range(dataset_size):
            if type == 'uniform':
                noise_set.append(torch.rand(noise_shape))

            elif type == 'gaussian' or type == 'normal':
                noise_set.append(torch.abs(torch.randn(noise_shape)).clamp(0, 1))

            else:
                raise ValueError("Unsupported noise type. Use 'uniform' or 'gaussian'/'normal'.")

        return torch.stack(noise_set, dim=0)

    def get_noise(self) -> torch.Tensor:
        if self.noise_set is None:
            raise ValueError("Noise is not generated. Call generate_noise() first.")

        return self.noise_set

    def augment_dataset(self, noise_set: torch.Tensor) -> torch.Tensor:
        augmented_dataset = []

        print("Augmenting dataset...")
        for i in tqdm(range(self.samples.shape[0])):
            augmented_dataset.append(self.augment_sample(self.samples[i], noise_set[i], self.aug_indices))
        self.aug_samples = torch.stack(augmented_dataset, dim=0)

    def augment_sample(self, sample_tensor: torch.Tensor, noise_tensor: torch.Tensor, aug_indices: list[int]) -> torch.Tensor:
        aug_shape = sample_tensor.shape
        aug_flat_size = aug_shape[1] * aug_shape[2]

        sample_shape = sample_tensor.shape
        noise_shape = noise_tensor.shape
        aug_shape = (sample_shape[0], sample_shape[1] + noise_shape[1], sample_shape[2] + noise_shape[2])

        aug_sample = torch.zeros(aug_shape)

        for c in range(aug_shape[0]):
            sample_flat = torch.flatten(sample_tensor[c])
            noise_flat = torch.flatten(noise_tensor[c])
            aug_flat = torch.flatten(aug_sample[c])

            sample_iter = iter(sample_flat)
            noise_iter = iter(noise_flat)

            for i in range(aug_flat_size):
                if i in aug_indices[c]:
                    aug_flat[i] = next(noise_iter)
                else:
                    aug_flat[i] = next(sample_iter)

            return aug_sample.reshape(aug_shape)

    def save_augmented_dataset(self, save_path: str) -> None:
        if self.aug_samples is None:
            raise ValueError("Augmented samples are not available. Call augment_dataset() first.")

        data_to_save = {
            'images': self.aug_samples,
            'labels': self.labels
        }
        
        try:
            torch.save(data_to_save, save_path)
            print(f"Augmented dataset saved successfully to {save_path}.")
        except Exception as e:
            print(f"Failed to save augmented dataset to {save_path}: {e}")


def display_image(image_tensor: torch.Tensor):
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    
    image_np = image_tensor.detach().numpy().transpose((1, 2, 0))
    
    image_np = np.clip(image_np, 0, 1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title('Image Display')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dataset Obfuscator')

    parser.add_argument('--trainset', type=str, default='../datasets/cifar10_train.pt',
                        help='Path to train dataset')
    parser.add_argument('--testset', type=str, default='../datasets/cifar10_test.pt',
                        help='Path to test/validation dataset')

    # args = parser.parse_args()

    # samples, labels = load_dataset(args.testset)

    # noise_set = generate_noise(samples.shape, 0.25, 'uniform')
    # noise_size = noise_set.shape[1] * noise_set.shape[2]

    # aug_indices = get_random_aug_indices(samples.shape[1:], noise_set.shape)

    # aug_sample = augment_sample(samples[0], noise_set[0], aug_indices)
    # print(aug_sample.shape)

    # print(samples[0][0])

    # print(torch.randn(sample_shape))
    # train_data = load_dataset(args.trainset)

    dataset_obfuscator = DatasetObfuscator('../datasets/cifar10_test.pt')
    noise = dataset_obfuscator.generate_noise(0.25, 'uniform')
    dataset_obfuscator.set_random_aug_indices(noise.shape)
    dataset_obfuscator.augment_dataset(noise)
    aug_sample = dataset_obfuscator.aug_samples[0]
    print(aug_sample.shape)