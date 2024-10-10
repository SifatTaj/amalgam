import collections
import torch
from itertools import repeat
from typing import Optional, List, Tuple, Union, TypeVar

from torch import Tensor
import torch.nn.functional as F
from torch._torch_docs import reproducibility_notes
from torch.nn.modules.conv import _ConvNd, convolution_notes

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import math

import time
from multiprocessing import Pool


# Function to augment a single input

def aug_input(data):
    img = data[0]

    aug_tensors = torch.zeros(num_channel, aug_tensor_size)
    aug_img_shape = (int(math.sqrt(aug_tensor_size)), int(math.sqrt(aug_tensor_size)))
    aug_img = torch.zeros(num_channel, aug_img_shape[0], aug_img_shape[1])

    for c in range(num_channel):
        img_flat = torch.reshape(img[c], (-1,))
        j = 0
        for i in range(aug_tensor_size):
            if i in aug_indices[c]:
                aug_tensors[c][i] = 1
            else:
                aug_tensors[c][i] = img_flat[j]
                j += 1

        aug_img[c] = torch.reshape(aug_tensors[c], aug_img_shape)

    return aug_img, data[1]


if __name__ == '__main__':

    aug_percentages = [1.0]

    for aug_percentage in aug_percentages:

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load original datasets
        cifar_trainset = datasets.CIFAR100(root='data_torch',
                                          train=True,
                                          transform=transform_train,
                                          download=True)

        cifar_testset = datasets.CIFAR100(root='data_torch',
                                         train=False,
                                         transform=transform_test,
                                         download=True)

        # Calculate augmentation parameters
        img = cifar_testset[10][0]
        num_channel = img.shape[0]
        img_dim = img.shape[2]
        aug_persent = aug_percentage  # Set percentage
        aug_dim = int(img_dim * aug_persent)
        aug_tensor_dim = img_dim + aug_dim

        aug_tensor_size = aug_tensor_dim * aug_tensor_dim
        aug_size = aug_tensor_size - (img_dim * img_dim)

        aug_indices = []
        aug_trainset = []
        aug_testset = []

        for i in range(num_channel):
            aug_indices.append(np.random.choice(np.arange(0, aug_tensor_size), replace=False, size=aug_size))

        aug_indices = np.array(aug_indices)

        # Augment trainset
        start = time.time()

        with Pool(20) as pool:
            # pool.map(aug_pool_split_train, trainset_split)
            results = pool.map(aug_input, cifar_trainset)
            aug_trainset.append(results)

        trainset_time = time.time() - start

        # Augment testset
        start = time.time()

        with Pool() as pool:
            # pool.map(aug_pool_split_test, testset_split)
            results = pool.map(aug_input, cifar_testset)
            aug_testset.append(results)

        testset_time = time.time() - start

        print('Augmentation time:', testset_time + trainset_time)
        print('trainset_time time:', trainset_time)
        print('testset_time time:', testset_time)

        aug_trainset_np = np.array(aug_trainset[0])
        aug_testset_np = np.array(aug_testset[0])

        np.save(f'aug_datasets/aug_multi_testset.npy_{aug_percentage}', aug_testset_np)
        np.save(f'aug_datasets/aug_multi_trainset.npy_{aug_percentage}', aug_trainset_np)
        np.save(f'aug_datasets/aug_multi_indices.npy_{aug_percentage}', aug_indices)

        f = open(f"aug_datasets/aug_multi_time_{aug_percentage}.txt", "a")
        f.write(f'\nAugmentation time: {testset_time + trainset_time}')
        f.write(f'\ntrainset_time time: {trainset_time}')
        f.write(f'\ntestset_time time: {testset_time}')
        f.close()