# LeNet-1
Implementation of the Convolutional neural network (LeNet-1) described in the paper [Backpropagation Applied to Handwritten Zip Code Recognition](https://ieeexplore.ieee.org/document/6795724) in PyTorch.


![image](res/architecture.png)

## Usage

To train the model, just run:
```bash
python3 train.py
```

Creating the model:
```python
from lenet_1989.lenet1989 import LeNet1989


# creating the model
net = LeNet1989()

# forward pass
y_pred = net.forward(x)

# printing model stats
print(net)
```

```bash
Stats from LeNet-1
total units:              1256
total connections:        64660
independent parameters:   9760
```


## Results

Results of the paper after 23 passes:

```bash
pass: 23
train report - loss: 0.00250     error: 0.0014   missclassifications: 10
test  report - loss: 0.01800     error: 0.0500   missclassifications: 102
```
My results after 23 passes:

```bash
pass: 23
train report - loss: 0.00101    error: 0.00521  missclassifications: 38
test  report - loss: 0.00811    error: 0.04933  missclassifications: 99
```

These results match pretty much the results from the original paper. Maybie with some hyperparameter optimization (for the best learning rate) we could achive better results.

## Notes

Im really to sure, that this is the actual "LeNet-1", but wikipedia says so.

**"In 1989, Yann LeCun et al. at Bell Labs first applied the backpropagation algorithm to practical applications, and believed that the ability to learn network generalization could be greatly enhanced by providing constraints from the task's domain. He combined a convolutional neural network trained by backpropagation algorithms to read handwritten numbers and successfully applied it in identifying handwritten zip code numbers provided by the US Postal Service. This was the prototype of what later came to be called LeNet-1.[3]"**

They used *"9298 segmented numerals digitized from handwritten zip codes that appeared on U.S. mail passing through the Buffalo, NY post office. "*, i couldn't find this dataset in the internet, so i simulated it using MNIST.

*"For units in layer H1 that are one unit apart, theier receptive fields (in the input layer) are two pixels apart"* $ \rightarrow $ `Stride = 2` (same from H1 to H2)

1 Unit in H2.X with $X \in \{1, ..., 12\}$ has `8 * 5 * 5 = 200` inputs from eight of the $8 \times 8$ feature maps via the $5 \times 5$ kernels

Not clear explanation how to select 8 of the 12 feature maps between H1 and H2, i did it like @karpathy

No information about padding

No information about used hyperparameters (i.e. learning rate)

No information how the bias was initialized (assumed to be zero)

They used mean squared error as a objective, instead of cross-entropy

I one hot encoded the targets, because of the MSE objective

This Convolutional Neural Network is often called LeNet-1

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