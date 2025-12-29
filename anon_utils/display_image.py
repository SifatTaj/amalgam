import torch
import numpy as np
import matplotlib.pyplot as plt

def display_image(image_tensor: torch.Tensor):
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()
    
    image_np = image_tensor.detach().numpy().transpose((1, 2, 0))
    
    image_np = np.clip(image_np, 0, 1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.set_title('Image Display')
    ax.axis('off')
    plt.tight_layout()
    plt.show()