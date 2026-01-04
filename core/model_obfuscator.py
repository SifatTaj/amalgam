import torch
from core.custom_conv import AnonConv2d

"""
Model obfuscator for PyTorch models.

Currently supports CNN architectures.
"""
class ModelObfuscator:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.original_first_conv = None

    def replace_first_conv_layer(self, aug_indices: list, deanon_dim: int) -> None:

        if hasattr(self.model, 'conv1'):
            original_conv = self.model.conv1
            self.original_first_conv = original_conv
            in_channels = original_conv.in_channels
            out_channels = original_conv.out_channels
            kernel_size = original_conv.kernel_size
            stride = original_conv.stride
            padding = original_conv.padding
            bias = original_conv.bias is not None

            new_conv = AnonConv2d(in_channels, out_channels, kernel_size,
                                    aug_indices=aug_indices, deanon_dim=deanon_dim,
                                       stride=stride, padding=padding, bias=bias)

            self.model.conv1 = new_conv
        else:
            raise ValueError("The model does not have a 'conv1' layer to replace.")

    def augment_layer(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError("Layer augmentation not implemented yet.")

    def get_obfuscated_model(self) -> torch.nn.Module:
        return self.model

    def deobfuscate_model(self) -> torch.nn.Module:
        if self.original_first_conv is None:
            raise ValueError("No original first convolutional layer stored.")

        trained_conv = self.model.conv1

        with torch.no_grad():
            self.original_first_conv.weight.copy_(trained_conv.weight)
            if trained_conv.bias is not None and self.original_first_conv.bias is not None:
                self.original_first_conv.bias.copy_(trained_conv.bias)

        self.model.conv1 = self.original_first_conv
        return self.model