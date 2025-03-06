"""
Author: Raphael Senn

Syntax:
    mc          : missclassification
    mcr         : missclassification rate 
    pass        : epoch
    mse_train   : mean squarred error on training set
    mse_test    : mean squarred error on test set
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lenet1.create_data import create_data
from lenet1.lenet1 import LeNet1


def evaluate(
        model: nn.Module,
        dataloader: DataLoader
        ) -> tuple[float, float, float]:

    # set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device 

    # init criterion
    criterion = nn.MSELoss()
    
    total_loss = 0
    total_error = 0
    
    # disable gradient calculations
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # move batches to same device as model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # one hot encoding for y_batch
            y_batch_hot = F.one_hot(y_batch, num_classes=10).float()

            # predict
            y_pred = model.forward(X_batch)
            
            # calculate MSE loss
            loss = criterion(y_pred, y_batch_hot)
            total_loss += loss.item()

            # calculate error rate
            y_pred = torch.argmax(y_pred, keepdim=True)
            if (y_batch != y_pred): total_error += 1

    return total_loss / len(dataloader), total_error / len(dataloader), total_error


def train(
        model: nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: DataLoader,
        lr: float,
        passes: int,
        device: str,
        verbose: bool
        ) -> None:

    # set model to train mode 
    model.train()

    # move model to correct device
    model.to(device)

    # init optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=lr)

    # init objective
    criterion = nn.MSELoss()

    for pass_n in range(passes):
        for X_batch, y_batch in dataloader_train:
            # move batches to same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # one hot encoding y_batch (for mse)
            y_batch_hot = F.one_hot(y_batch, num_classes=10).float()

            # clear previous gradients
            optimizer.zero_grad()

            # predict
            y_pred = model.forward(X_batch)

            # calculate MSE loss
            loss = criterion(y_pred, y_batch_hot)

            # backpropagation
            loss.backward()

            # update model parameters
            optimizer.step()

        # calculate performance
        mse_train, mcr_train, mc_train = evaluate(model, dataloader_train)
        mse_test, mcr_test, mc_test = evaluate(model, dataloader_test)
        
        # print performance report if verbose
        if verbose:
            report =  f'pass: {pass_n+1}\n'
            report += f'train report - loss: {mse_train:.5f}\terror: {mcr_train:.5f}\tmissclassifications: {mc_train}\n'
            report += f'test  report - loss: {mse_test:.5f} \terror: {mcr_test:.5f} \tmissclassifications: {mc_test}\n'
            print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.1725,
        help='Learning rate for stochastic gradient descent'
    )
    parser.add_argument(
        '-p',
        '--training_passes',
        type=int,
        default=23,
        help='Number of training passes'
    )
    parser.add_argument(
        '-verbose',
        '--verbose',
        type=bool,
        default=True,
        help='Printing: mse loss, error rate and number of missclassifications while training after each pass (measured on the entire training and test set)'
    )
    parser.add_argument(
        '-seed',
        '--seed',
        type=int,
        default=42,
        help='Seed for reproducibility'
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='cpu',
        help='Computing device: CPU or CUDA'
    )
    args = parser.parse_args()

    # load training and testing data
    dataloader_train, dataloader_test = create_data(seed=args.seed) 

    # set seed (to all devices, both CPU and CUDA)
    torch.manual_seed(args.seed)

    # create the model
    lenet = LeNet1()

    # printing model stats
    print(lenet)

    # start training
    train(
        model=lenet,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        lr=args.learning_rate,
        passes=args.training_passes,
        device=args.device,
        verbose=args.verbose
    )