import torchvision
import torchvision.transforms as transforms
import argparse

def download_cifar10(path):
    
    # Define transformations for training and testing sets
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

    # Download CIFAR-10 training and test datasets
    torchvision.datasets.CIFAR10(root='datasets',
                                    train=True,
                                    transform=transform_train,
                                    download=True)

    torchvision.datasets.CIFAR10(root='datasets',
                                    train=False,
                                    transform=transform_test,
                                    download=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download CIFAR-10 Dataset')
    parser.add_argument('--path', type=str, default='datasets',
                        help='Path to download the CIFAR-10 dataset')
    args = parser.parse_args()
       
    download_cifar10(args.path)