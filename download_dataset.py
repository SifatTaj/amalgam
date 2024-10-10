import torchvision
import torchvision.transforms as transforms

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

torchvision.datasets.CIFAR10(root='datasets',
                                  train=True,
                                  transform=transform_train,
                                  download=True)

torchvision.datasets.CIFAR10(root='datasets',
                                 train=False,
                                 transform=transform_test,
                                 download=True)