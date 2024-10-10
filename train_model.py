import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.anon_resnet import AnonResNet18
from anon_utils.param_count_table import count_total_params, show_parameters

from models.original_resnet import ResNet18
from utils import progress_bar

from anon_utils.anon_dataset_loader import AugDataset2


def train(net, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print('label: ', classes[targets[0]])
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_losses.append(train_loss)
    train_accuracy.append(100. * correct / total)


def test(net, epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / total)
    return 100. * correct / total


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

original_trainset = torchvision.datasets.CIFAR10(root='datasets',
                                                  train=True,
                                                  transform=transform_train,
                                                  download=True)

original_testset = torchvision.datasets.CIFAR10(root='datasets',
                                                 train=False,
                                                 transform=transform_test,
                                                 download=True)

original_testloader = torch.utils.data.DataLoader(
    original_testset, batch_size=128, shuffle=True, num_workers=2)

original_trainloader = torch.utils.data.DataLoader(
    original_trainset, batch_size=128, shuffle=True, num_workers=2)


ds_name = 'CIFAR10'
model_name = 'resnet'

aug_percentages = [0.0, 0.25]
aug_indices_all = torch.load(f'aug_datasets/{ds_name}_indices.pt')

for level_idx, aug_percent in enumerate(aug_percentages):

    # Data
    print('==> Preparing data..')
    if aug_percent != 0:
        aug_trainset = torch.load(f'aug_datasets/{ds_name}_train_{aug_percent}.pt')
        aug_trainset = list(aug_trainset.parameters())[0]
        aug_trainset = AugDataset2(aug_trainset, original_trainset)
        aug_trainloader = torch.utils.data.DataLoader(aug_trainset, batch_size=128, shuffle=True, num_workers=2)

        aug_testset = torch.load(f'aug_datasets/{ds_name}_test_{aug_percent}.pt')
        aug_testset = list(aug_testset.parameters())[0]
        aug_testset = AugDataset2(aug_testset, original_testset)
        aug_testloader = torch.utils.data.DataLoader(aug_testset, batch_size=128, shuffle=True, num_workers=2)

    deanon_dim = 28 if ds_name == 'MNIST' else 32
    num_channel = 1 if ds_name == 'MNIST' else 3
    num_classes = 100 if ds_name == 'CIFAR100' else 10

    # Hyper params
    learning_rate = 0.1
    num_epochs = 100

    # Aug index extraction
    if aug_percent != 0:
        aug_indices = []
        for c in range(num_channel):
            aug_index = aug_indices_all[level_idx - 1][c]
            aug_index = aug_index.numpy().astype(int)
            aug_index = aug_index[aug_index != 0]
            aug_indices.append(aug_index)

        aug_indices = np.array(aug_indices)

    # Model
    print('==> Building model..')
    if model_name == 'resnet':
        net = ResNet18(num_classes, num_channel) \
            if level_idx == 0 \
            else AnonResNet18(num_classes=num_classes, num_channel=num_channel, aug_indices=aug_indices, deanon_dim=deanon_dim, aug_percent=aug_percent)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    anon_net_params = count_total_params(net)

    train_losses = []
    train_accuracy = []

    test_losses = []
    test_accuracy = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start = time.perf_counter()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if aug_percent > 0:
            train(net, epoch, aug_trainloader)
            test(net, epoch, aug_testloader)
            scheduler.step()
        else:
            train(net, epoch, original_trainloader)
            test(net, epoch, original_testloader)
            scheduler.step()

    end = time.perf_counter() - start

    print(torch.cuda.memory_summary())

    actual_acc = test(net, 0, original_testloader)
    print('time taken:', end)

    np.save(f'results/{model_name}/train_losses_resnet_{aug_percent}_{ds_name}.npy', np.array(train_losses))
    np.save(f'results/{model_name}/test_losses{aug_percent}_{ds_name}.npy', np.array(test_losses), )
    np.save(f'results/{model_name}/test_accuracy{aug_percent}_{ds_name}.npy', np.array(test_accuracy), )
    np.save(f'results/{model_name}/train_accuracy{aug_percent}_{ds_name}.npy', np.array(train_accuracy), )

    f = open(f'results/{model_name}/{model_name}_result_{aug_percent}_{ds_name}.txt', 'a')
    f.write(f'\nnumber of anon params {anon_net_params}')
    f.write(f'\nnumber of original params {count_total_params(net)}')
    f.write(f'\ndeanon accuracy {actual_acc}')
    f.write(f'\nTraining time: {end}')
    f.write(f'\n\n{torch.cuda.memory_summary()}')