import torch
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

def test_cuda():
    obfs_data = torch.load("datasets/cifar10_test_obfuscated.pt")
    obfs_samples = obfs_data['images']
    labels = obfs_data['labels']
    aug_indices = obfs_data['aug_indices']

    batch_size = 4
    aug_indices_size = len(aug_indices[0])
    filtered_input_size = 32 * 32
    aug_input_size = obfs_samples.shape[2] * obfs_samples.shape[3]
    n_channels = obfs_samples.shape[1]

    sample_batch = obfs_samples[:batch_size].flatten().contiguous().numpy()
    filtered_input = np.zeros(batch_size * n_channels * filtered_input_size, dtype=np.float32)
    aug_indices = aug_indices.flatten().contiguous().numpy()

    lib_path = "core/lib/filter_input.so"

    # Load the CUDA library
    lib = ctypes.CDLL(lib_path)

    '''
    void filter_aug_input(
        const float* aug_input,
        float* filtered_input,
        const int* aug_indices,
        const int aug_indices_size,
        const int batch_size,
        const int aug_input_size,
        const int filtered_input_size,
        const int n_channels,
    ) {
    '''

    # Define the function signature
    lib.filter_aug_input.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    lib.filter_aug_input(
        sample_batch.astype(np.float32),
        filtered_input,
        aug_indices.astype(np.int32),
        ctypes.c_int(aug_indices_size),
        ctypes.c_int(batch_size),
        ctypes.c_int(aug_input_size),
        ctypes.c_int(filtered_input_size),
        ctypes.c_int(n_channels)
    )

    print(len(sample_batch))
    print(len(filtered_input))

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
    # test_model()
    test_cuda()