from core.dataset_obfuscator import DatasetObfuscator

def main():
    dataset_obfuscator = DatasetObfuscator('datasets/cifar10_test.pt')
    noise = dataset_obfuscator.generate_noise(0.25, 'uniform')
    dataset_obfuscator.set_random_aug_indices(noise.shape)
    dataset_obfuscator.augment_dataset(noise)
    aug_sample = dataset_obfuscator.aug_samples[0]
    print(aug_sample.shape)


if __name__ == "__main__":
    main()