# Implementation of ResNet50
---
This repository contains the paper implementation of `ResNet50`, a CNN-based architecture. The primary goal of this repository is to show how to implement this paper using `tensorflow-keras`. Model performance and accuracy are not taken into consideration while implementing this paper. I just focused on the implementation part.

For the implementation, I used the facial emotion dataset. The reason for not opting for other well-known datasets like MNIST is because those datasets are fairly simple. Whereas facial emotion recognition aka FER is a bit complex task to do.

## Paper Explanation
### The Problem
The problem is the degradation problem; as we increase the depth of the network, instead of the accuracy increasing, it drops. This drop isn’t just on the validation set data but also on the training data. As a result of this, overfitting can be ruled out as the cause of the problem because if the model was overfitting, the training accuracy, instead of dropping, would be high which is different from the observation.

<p align="center">
  <img width="850" height="400" src="https://mohitjainweb.files.wordpress.com/2018/06/degradation-problem1.png?w=768">
</p>

The above image shows the Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer plain networks.
In the paper, it is mentioned that the `vanishing gradient` has been taken care of by `normalized initialization` and `intermediate normalization layers` and the problem is due to because "_current solvers are unable to find a proper solution in feasible time"_. In other words, it’s an optimization problem. Somehow, it is becoming very difficult for a deep "plain" network to create a good mapping (a function) from the input images to the output labels.

### The Architecture
#### Residual Learning
Residual learning is based on the hypothesis, i.e, it is theoretically possible to map any function using multiple non-linear layers, however, practically not all mappings are easy and feasible to obtain.
Denoting the underlying (required) mapping as **_H(x)_**, the stacked non-linear layers fit another mapping, called residual mapping, of **_F(x) := H(x) – x_**. It is hypothesized that it is easier to fit the residual mapping, **_F(x)_**, than to optimise the original mapping, **_H(x) := F(x) + x_**. An example to see this is, say the ideal optimal mapping for **_H(x)_** is the identity function i.e. **_H(x) = x_**, then it will be easier to push the residual to zero, F(x) = 0, than to fit an identity mapping by a stack of non-linear layers.

#### ResNet Architecture
<p align="center">
  <img width="400" height="200" src="https://mohitjainweb.files.wordpress.com/2018/06/residual-block.png">
</p>

The figure shows the smallest building block of a ResNet. It is a couple of stacked layers (minimum two) with a skip connection. Skip connections are mainly just identity mappings and hence contribute no additional parameters. Residual learning is applied to these stacked layers. The block can be represented as:

**_y = F(x, {Wi}) + x_**

F is the residual function learned by the stacked layers. F + x is element-wise addition. The dimensions of x and F must be equal to perform the addition.

### Conclusion
From the 22 layer GoogLeNet to the monstrous 152 layer ResNet was a huge leap in just one year! ResNets achieved a 3.57% top-5 error on the ImageNet competition.


Link to the original paper: [Paper](https://arxiv.org/pdf/1512.03385.pdf)
