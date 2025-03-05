"""
Author: Raphael Senn

Syntax:
    mc          : missclassification
    mcr         : missclassification rate 
    pass        : epoch
    mse_train   : mean squarred error on training set
    mse_test    : mean squarred error on test set
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lenet_1989.create_data import create_data
from lenet_1989.lenet1989 import LeNet1989


def evaluate(
        model: nn.Module,
        dataloader: DataLoader
        ) -> tuple[float, float, float]:

    # set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device 

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
            loss = torch.mean((y_pred - y_batch_hot)**2)
            total_loss += loss.item()

            # calculate error rate
            y_pred = torch.argmax(y_pred, keepdim=True)
            if (y_batch != y_pred): total_error += 1

    return total_loss / len(dataloader), total_error / len(dataloader), total_error


def train(
        model: nn.Module,
        dataloader_train: DataLoader,
        dataloader_test: DataLoader,
        passes: int=23,
        verbose: bool=True,
        return_mse_mcr: bool=False
        ) -> None | tuple:

    # cache mse_loss (train, test) and error rate (train, test)
    mse_mcr_train= []
    mse_mcr_test = []

    # set model to train mode 
    model.train()

    # set device to cpu or gpu (if cuda is available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # init optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.195)

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

        if return_mse_mcr:
            mse_mcr_train.append((mse_train, mcr_train))
            mse_mcr_test.append((mse_test, mcr_test))
    if return_mse_mcr: return (mse_mcr_train, mse_mcr_test)


if __name__ == '__main__':
    # load training and testing data
    dataloader_train, dataloader_test = create_data() 

    # create the model
    net = LeNet1989()

    # printing model stats
    print(net)

    # start training
    train(net, dataloader_train, dataloader_test)