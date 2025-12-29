import torchvision
import torchvision.transforms as transforms
import argparse
import torch
import tqdm

def download_cifar10(path: str):
    
    print(f"Downloading CIFAR-10 dataset to {path}...")
    # Define transformations for training and testing sets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='datasets',
                                    train=True,
                                    transform=transform_train,
                                    download=True)

    testset = torchvision.datasets.CIFAR10(root='datasets',
                                    train=False,
                                    transform=transform_test,
                                    download=True)

    print("Processing training set...")
    trainset_images = []
    trainset_labels = []

    for image_tensor, label in tqdm.tqdm(trainset):
        trainset_images.append(image_tensor)
        trainset_labels.append(label)

    trainset_data = torch.stack(trainset_images)

    testset_images = []
    testset_labels = []

    print("Processing test set...")
    for image_tensor, label in tqdm.tqdm(testset):
        testset_images.append(image_tensor)
        testset_labels.append(label)

    testset_data = torch.stack(testset_images)
    testset_labels = torch.tensor(testset_labels)

    trainset_data = torch.stack(trainset_images)
    trainset_labels = torch.tensor(trainset_labels)

    torch.save({'images': testset_data, 'labels': testset_labels}, f"{path}/cifar10_test.pt")
    torch.save({'images': trainset_data, 'labels': trainset_labels}, f"{path}/cifar10_train.pt")


def verify_download(path: str):
    try:
        data = torch.load(f"{path}/cifar10_test.pt")
        print("CIFAR-10 test dataset loaded successfully.")
        print(f"Test images shape: {data['images'].shape}")
        print(f"Test labels shape: {data['labels'].shape}")

        data = torch.load(f"{path}/cifar10_train.pt")
        print("CIFAR-10 train dataset loaded successfully.")
        print(f"Train images shape: {data['images'].shape}")
        print(f"Train labels shape: {data['labels'].shape}")

        print("CIFAR10 is now ready to use.")
    except Exception as e:
        print(f"Failed to load CIFAR-10 test dataset: {e}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download CIFAR-10 Dataset')
    parser.add_argument('--path', type=str, default='datasets',
                        help='Path to download the CIFAR-10 dataset')
    args = parser.parse_args()

    download_cifar10(args.path)
    verify_download(args.path)