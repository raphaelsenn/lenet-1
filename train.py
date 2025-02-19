import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset


def create_data() -> tuple[DataLoader, DataLoader]:

    # -------------------------------------------------------------------------
    # 1. Reshape to 16 x 16 images
    # 2. Convert to tensor
    # 3. Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    mnist_test = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transform,
                                               download=True)

    N_train = 7291
    N_test = 2007
    
    perm_train = np.random.permutation(N_train)
    perm_test = np.random.permutation(N_test)

    mnist_train = Subset(mnist_train, perm_train)
    mnist_test = Subset(mnist_test, perm_test)

    dataloader_train = DataLoader(mnist_train, shuffle=True)
    dataloader_test = DataLoader(mnist_test, shuffle=True)

    return dataloader_train, dataloader_test


if __name__ == '__main__':
    data_train, data_test = create_data()
    print(len(data_train.dataset))
    print(len(data_test.dataset))