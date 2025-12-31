import torch

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

# print("Original Sample 0:")
# print(orig_sample_0)
# print("Obfuscated Sample 0:")
# print(obfs_sample_0)