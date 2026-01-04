from core.dataset_obfuscator import DatasetObfuscator
from core.model_obfuscator import ModelObfuscator
from models.resnet import ResNet18

def dataset_obfuscation():
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

def model_obfuscation():
    # Load obfuscated dataset
    obfs_data = torch.load("datasets/cifar10_test_obfuscated.pt")
    obfs_samples = obfs_data['images']
    labels = obfs_data['labels']
    aug_indices = obfs_data['aug_indices']

    # Initialize original model
    model = ResNet18(num_classes=10, num_channel=3)

    # Initialize model obfuscator
    model_obfuscator = ModelObfuscator(model)
    model_obfuscator.replace_first_conv_layer(aug_indices=aug_indices, deanon_dim=(32, 32))

    # Get obfuscated model
    obfs_model = model_obfuscator.get_obfuscated_model()

    # The obfuscated model can now be trained on obfs_samples and labels

    # After training, deobfuscate the model
    deobfuscate_mode = model_obfuscator.deobfuscate_model()

if __name__ == "__main__":
    dataset_obfuscation()
    model_obfuscation()