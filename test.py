import torch

def test_dataset():
    obfs_data = torch.load("datasets/cifar10_test_obfuscated.pt")
    obfs_samples = obfs_data['images']
    labels = obfs_data['labels']
    aug_indices = obfs_data['aug_indices']

    original_data = torch.load("datasets/cifar10_test.pt")
    original_samples = original_data['images']

    print(original_samples.shape)
    print(obfs_samples.shape)
    print(aug_indices.shape)

    orig_sample_0 = [m.item() for m in original_samples[0][0].flatten()]

    obfs_sample_0 = [m.item() for m in obfs_samples[0][0].flatten()]
    deobfs_sample_0 = []

    for i in range(len(obfs_sample_0)):
        if i not in aug_indices[0]:
            deobfs_sample_0.append(obfs_sample_0[i])

    print(len(orig_sample_0))
    print(len(deobfs_sample_0))

    print("Deobfuscated matches original:", orig_sample_0 == deobfs_sample_0)

    print("Original Sample 0:")
    print(orig_sample_0)
    print("Obfuscated Sample 0:")
    print(obfs_sample_0)

from models.resnet import ResNet18
from core.model_obfuscator import ModelObfuscator

def test_model():

    obfs_data = torch.load("datasets/cifar10_test_obfuscated.pt")
    obfs_samples = obfs_data['images']
    labels = obfs_data['labels']
    aug_indices = obfs_data['aug_indices']

    print(type(aug_indices))

    model = ResNet18(num_classes=10, num_channel=3)

    model_obfuscator = ModelObfuscator(model)
    model_obfuscator.replace_first_conv_layer(aug_indices=aug_indices, deanon_dim=(32, 32))
    obfs_model = model_obfuscator.get_obfuscated_model()
    obfs_model = obfs_model.to('cuda')
    obfs_samples = obfs_samples.to('cuda')

    out = obfs_model(obfs_samples[:4])
    print("Output shape:", out.shape)

    deobfuscate_mode = model_obfuscator.deobfuscate_model()

if __name__ == "__main__":
    # test_dataset()
    test_model()