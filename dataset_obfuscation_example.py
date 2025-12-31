from core.dataset_obfuscator import DatasetObfuscator

def main():
    # initialize dataset obfuscator for train dataset
    testset_obfuscator = DatasetObfuscator('datasets/cifar10_train.pt', amount=0.25)

    # generate uniform noise for 25% augmentation
    noise = testset_obfuscator.generate_noise(0.25, 'uniform')

    # generate random augmentation indices.
    aug_indices = testset_obfuscator.generate_random_indices(noise.shape)

    # set the generated augmentation indices
    testset_obfuscator.set_random_aug_indices(aug_indices)

    # augment the dataset with the generated noise
    testset_obfuscator.augment_dataset(noise)

    # save the obfuscated train dataset
    testset_obfuscator.save_augmented_dataset('datasets/cifar10_train_obfuscated.pt')

    # initialize dataset obfuscator for test dataset
    trainset_obfuscator = DatasetObfuscator('datasets/cifar10_test.pt', amount=0.25)

    # generate uniform noise for 25% augmentation
    noise = trainset_obfuscator.generate_noise(0.25, 'uniform')

    # set trainset augmentation indices to be the same as testset
    trainset_obfuscator.set_random_aug_indices(aug_indices)

    # augment the dataset with the generated noise
    trainset_obfuscator.augment_dataset(noise)

    # save the obfuscated test dataset
    trainset_obfuscator.save_augmented_dataset('datasets/cifar10_test_obfuscated.pt')

if __name__ == "__main__":
    main()