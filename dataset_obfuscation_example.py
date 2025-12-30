from core.dataset_obfuscator import DatasetObfuscator

def main():
    testset_obfuscator = DatasetObfuscator('datasets/cifar10_test.pt')
    noise = testset_obfuscator.generate_noise(0.25, 'uniform')
    aug_indices = testset_obfuscator.generate_random_indices(noise.shape)
    testset_obfuscator.set_random_aug_indices(aug_indices)
    testset_obfuscator.augment_dataset(noise)
    testset_obfuscator.save_augmented_dataset('datasets/cifar10_test_obfuscated.pt')

    trainset_obfuscator = DatasetObfuscator('datasets/cifar10_train.pt')
    noise = trainset_obfuscator.generate_noise(0.25, 'uniform')
    trainset_obfuscator.set_random_aug_indices(aug_indices)
    trainset_obfuscator.augment_dataset(noise)
    trainset_obfuscator.save_augmented_dataset('datasets/cifar10_train_obfuscated.pt')

if __name__ == "__main__":
    main()