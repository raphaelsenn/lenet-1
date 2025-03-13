# LeNet-1
Implementation of the convolutional neural network (LeNet-1) described in the paper [Backpropagation Applied to Handwritten Zip Code Recognition](https://ieeexplore.ieee.org/document/6795724) in PyTorch.

![image](res/architecture.png)

## Usage

#### Automatic training

Just run:
```bash
python3 train.py
```

#### Manual training

Creating the data:
```python
import torch
from create_data import create_data


# set random seed
seed = 42
torch.manual_seed(seed)

# create training and testing data
dataloader_train, dataloader_test = create_data(seed=seed)
```

Creating the model:
```python
from lenet1.lenet1 import LeNet1


# creating the model
lenet = LeNet1()

# forward pass
y_pred = lenet.forward(x)

# printing model stats
print(lenet)
```

```bash
Stats LeNet-1
total units:              1256
total connections:        64660
independent parameters:   9760
```

Start training:
```python
from train import train


# start training
train(
  lenet,              # model
  dataloader_train,   # training data
  dataloader_test,    # test data
  0.15,               # learning rate
  23,                 # training passes
  'cpu',              # device
  True                # printing mse, error rate, ... while training
)
```

## Results

Results of the paper after 23 passes:

```text
pass: 23
train report - loss: 0.00250     error: 0.0014   missclassifications: 10
test  report - loss: 0.01800     error: 0.0500   missclassifications: 102
```
My results after 23 passes:

```text
pass: 23
train report - loss: 0.00101    error: 0.00521  missclassifications: 38
test  report - loss: 0.00811    error: 0.04933  missclassifications: 99
```

These results match pretty much the results from the original paper. Maybie with some hyperparameter optimization (i.e. for the best learning rate) we could achive much better results.

## Notes

#### About the Convolutional Neural Network

* "LeNet-1" consists of three hidden layers (H1 to H3), and one output layer

* Input is a $(16 \times 16)$ greyscale image (range between $[-1, 1]$), resulting in $16 * 16 = 256$ input neurons

* *"For units in layer H1 that are one unit apart, their receptive fields (in the input layer) are two pixels apart.*" $ \implies $ `stride=2` (same between layer $H1$ and $H2$)


##### Layer H1

* Layer $H1$ uses $12$ $(5 \times 5)$-kernels resulting in $12$ feature maps H1.1, ..., H1.12 where each feature map has a ($8 \times 8$) shape

* The $12$ $(12 \times 12)$-kernels result in $12 * 5 * 5 = 300$ learnable parameters (weights)

* Each Unit in $H1.X$ with $X \in \{1, ..., 12\}$ has its own bias (Conv2D in PyTorch uses $1$-bias for each feature map instead), resulting in $12 * 8 * 8 = 768$ biases

* Therefore layer $H1$ consists of $300 + 768 = 1068$ learnable parameters

##### Layer H2

* Layer $H2$ features $12$ feature maps, each feature map consists of $8$ $(5 \times 5)$-kernels, resulting in $12 * 8 * 5 * 5 = 2400$ learnable parameteres (weight) 

* Each unit in $H2$ combines local information coming from $8$ of the $12$ different feature maps in $H1$

* There is **NO** clear explanation how to select 8 of the 12 feature maps between layer $H1$ and $H2$, i did it like [@karpathy](https://github.com/karpathy)

* Each unit in $H2.X$ with $X \in \{1, ..., 12\}$ has $8 * 5 * 5 = 200$ inputs coming from $8$ $(8x8)$ feature maps (from H1) **AND** $1$-bias

* $H2$ consists of $12 * 4 * 4 = 192$-biases

* Therefore layer $H2$ consists of $2400 + 192 = 2592$ learnable parameters

##### Layer H3

* Layer $H3$ is fully connected to $H2$ (the $12$ feature maps H2.1, ..., H2.12 which are $12 * 4 * 4 = 192$ units)

* $H3$ consists of $30$ units and biases, resulting in $192 * 30 + 30=5790$ learnable parameters 

##### Output layer

* The output layer is fully-connected to layer $H3$

* The output layer consists of $10$ units and biases, resulting in $30 * 10 + 10 = 310$ learnable parameters

##### More notes about the neural net

* No information about padding

* No information about used hyperparameters (i.e. learning rate)

* No information how the biases were initialized (assumed to be zero)

* They used the mean squared error as an objective, instead of cross-entropy

* I one hot encoded the targets (during training), because of the MSE objective

#### About the data

* They used *"9298 segmented numerals digitized from handwritten zip codes that appeared on U.S. mail passing through the Buffalo, NY post office. "*

* I couldn't find this dataset in the internet, so i simulated it using MNIST.

## Insights about the Data

#### Viewing some random numbers

![image](res/random_numbers.png)

#### Number distribution

![image](res/number_distribution.png)

## Citations

```bibtex
@article{LeCun1989,
  title     = {Backpropagation Applied to Handwritten Zip Code Recognition},
  author    = {Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel},
  journal   = {Neural Computation},
  year      = {1989},
  publisher = {MIT Press}
}
```