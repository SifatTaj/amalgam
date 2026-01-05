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

    device = 'cuda:0'

    # Keep tensors on GPU
    sample_batch = obfs_samples[:batch_size].flatten().to(device)
    filtered_input = torch.zeros(batch_size * n_channels * filtered_input_size, dtype=torch.float32).to(device)
    # aug_indices is (n_channels, aug_indices_size), flatten to pass to kernel
    aug_indices_gpu = aug_indices.flatten().type(torch.int32).to(device)

    print(aug_indices_gpu.shape)
    print(aug_indices_gpu.dtype)
    lib_path = "core/lib/filter_input.so"

    # Load the CUDA library
    lib = ctypes.CDLL(lib_path)

    '''
    void filter_aug_input(
        float* aug_input,
        float* filtered_input,
        int* aug_indices,
        int aug_indices_size,
        int batch_size,
        int aug_input_size,
        int filtered_input_size,
        int n_channels
    ) {
    '''

    # Define the function signature using c_void_p for device pointers
    lib.filter_aug_input.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    lib.filter_aug_input(
        ctypes.c_void_p(sample_batch.data_ptr()),
        ctypes.c_void_p(filtered_input.data_ptr()),
        ctypes.c_void_p(aug_indices_gpu.data_ptr()),
        ctypes.c_int(aug_indices_size),
        ctypes.c_int(batch_size),
        ctypes.c_int(aug_input_size),
        ctypes.c_int(filtered_input_size),
        ctypes.c_int(n_channels)
    )

    # for i in range(aug_indices_size * 0, aug_indices_size * 1):
    #     print('after:', aug_indices_gpu[i])

    # idx = 11
    # channel_idx = idx % n_channels
    # aug_indices_offset = channel_idx * aug_indices_size

    # for j in range(aug_indices_size):
    #     aug_index = aug_indices_gpu[aug_indices_offset + j]
    #     print(f'aug_indices[{aug_indices_offset + j}] = {aug_index}')

    original_data = torch.load("datasets/cifar10_test.pt")
    original_samples = original_data['images']

    orig_sample_batch = original_samples[:batch_size].flatten().cuda()
    print(filtered_input)
    print(orig_sample_batch)
    match = filtered_input == orig_sample_batch
    print(torch.sum(~match))



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
    obfs_model = obfs_model.to('cuda:0')
    obfs_samples = obfs_samples.to('cuda:0')

    out = obfs_model(obfs_samples[:4])
    print("Output shape:", out.shape)

    deobfuscate_mode = model_obfuscator.deobfuscate_model()

if __name__ == "__main__":
    # test_dataset()
    test_model()
    # test_cuda()