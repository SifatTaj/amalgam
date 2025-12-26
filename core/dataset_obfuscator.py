import torch
import torchvision

def load_dataset(path):
    # Load a PyTorch dataset from the specified path
    dataset = torch.jit.load(path)
    return dataset

if __name__ == "__main__":
    dataset_path = "path/to/dataset.pt"
    dataset = load_dataset(dataset_path)
    print("Dataset loaded successfully.")