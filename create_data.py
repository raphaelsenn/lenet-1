"""
Author: Raphael Senn
"""
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def create_data(
        N_train:int=7291,
        N_test: int=2007,
        seed:int=42
        ) -> tuple[DataLoader, DataLoader]:
    """ 
    Data preprocessing:
        ToTensor                Images to Tensor and [0, 255] -> [0, 1]
        Resize                  28 x 28 -> 16 x 16
        Normalize(mean, std)    [0, 1] -> [-1, 1]
    """ 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((16, 16)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST_train dataset
    mnist_train = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    # Load MNIST_test dataset
    mnist_test = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transform,
                                               download=True)
    
    # Create 'random' permutations
    np.random.seed(seed)
    perm_train = np.random.permutation(N_train)
    perm_test = np.random.permutation(N_test)

    # Create handwritten zipcode datasets as subsets of MNIST
    mnist_train = Subset(mnist_train, perm_train)
    mnist_test = Subset(mnist_test, perm_test)

    dataloader_train = DataLoader(mnist_train, shuffle=True, batch_size=1)
    dataloader_test = DataLoader(mnist_test, shuffle=True, batch_size=1)

    return dataloader_train, dataloader_test