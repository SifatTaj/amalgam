import collections
import torch
from itertools import repeat
from typing import Optional, List, Tuple, Union, TypeVar

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

import numpy as np
import ctypes
import os.path

'''
`AnonConv2d`: A custom convolutional layer that can handle obfuscated inputs by deobfuscating them
using provided augmentation indices and a specified deobfuscation dimension.

'''

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


class AnonConv2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            aug_indices: torch.Tensor = None,
            deanon_dim: torch.Size =None,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.deanon_dim = deanon_dim
        self.aug_indices = aug_indices
        super(AnonConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        if input.shape == (input.shape[0], input.shape[1], self.deanon_dim[0], self.deanon_dim[1]):
                return self._conv_forward(input, self.weight, self.bias)

        lib_path = "core/lib/filter_input.so"

        if os.path.isfile(lib_path):

            lib = ctypes.CDLL(lib_path)
            device = input.device

            batch_size = input.shape[0]
            n_channels = input.shape[1]
            aug_input_size = input.shape[2] * input.shape[3]
            deanon_input_size = self.deanon_dim[1] * self.deanon_dim[0]
            aug_indices_size = len(self.aug_indices[0])

            aug_input = input.flatten().to(device)
            deanon_batch = torch.zeros(batch_size * n_channels * deanon_input_size, dtype=torch.float32).to(device)
            aug_indices_gpu = self.aug_indices.flatten().type(torch.int32).to(device)

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
                ctypes.c_void_p(aug_input.data_ptr()),
                ctypes.c_void_p(deanon_batch.data_ptr()),
                ctypes.c_void_p(aug_indices_gpu.data_ptr()),
                ctypes.c_int(aug_indices_size),
                ctypes.c_int(batch_size),
                ctypes.c_int(aug_input_size),
                ctypes.c_int(deanon_input_size),
                ctypes.c_int(n_channels)
            )

            deanon_batch = deanon_batch.reshape(batch_size, n_channels, self.deanon_dim[1], self.deanon_dim[0])
            return self._conv_forward(deanon_batch, self.weight, self.bias)

        else:

            device = input.device
            deanon_batch = torch.zeros(input.shape[0], input.shape[1], self.deanon_dim[0], self.deanon_dim[1], device=device)

            for idx, data in enumerate(input):
                aug_img_np = data.cpu().numpy()
                aug_img_np = aug_img_np.reshape(aug_img_np.shape[0], -1)

                deanon_img_list = []
                for c in range(data.shape[0]):
                    deanon_img_np = np.delete(aug_img_np[c], self.aug_indices[c])
                    deanon_img_np = deanon_img_np.reshape(self.deanon_dim[0], self.deanon_dim[1])
                    deanon_img_list.append(torch.from_numpy(deanon_img_np))

                deanon_batch[idx] = torch.stack(deanon_img_list).to(device)

            return self._conv_forward(deanon_batch, self.weight, self.bias)
