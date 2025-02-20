import numpy as np

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset


def create_handwritten_zipcodes(
        N_train:int=7291,
        N_test: int=2007,
        seed:int=42
        ) -> tuple[DataLoader, DataLoader]:

    # -------------------------------------------------------------------------
    # Creating 9298 handwritten zipcodes using MNIST 
    # -------------------------------------------------------------------------
    
    # 1. Reshape to 16 x 16 images
    # 2. Convert to tensor
    # 3. Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST train
    mnist_train = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    # Load MNIST test
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

    dataloader_train = DataLoader(mnist_train, shuffle=True)
    dataloader_test = DataLoader(mnist_test, shuffle=True)

    return dataloader_train, dataloader_test