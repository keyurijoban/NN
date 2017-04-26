---
title       : Neural Networks
subtitle    : Getting Started 
author      : Gunnvant Singh
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : [mathjax, quiz, bootstrap]            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
---

## Agenda

1. Neural Networks
 * Uses
 * Basic concepts: Perceptron, activations, loss functions
 * Multilayer perceptron
 * Gradient Descent and Back propagation
 * Convolutional Neural Networks
 * Transfer learning in CNN

2. Keras
 * Basic API
 * MLP implementation
 * CNN implementation

--- .class #id 

## Neural Networks : Uses

* Typical use case for Neural Networks is in the field of computer vision and NLP
* Traditional methods and analysis are not going to be replaced by neural networks
* This session will introduce Multi Layer Perceptron Model in the classification setting, aim: to make you self explorers.
* Current state of the art in Neural Networks: Convolutional Neural Networks, Recurrent Neural Networks


--- .class #id


## Basic Concepts

* Central Concepts:
 * Layers: Input, Hidden and Output
 * Activations
 * Cost functions

 

--- .class #id

## Basic Concepts 

* Layers: Input, Hidden and Output

<img src='mlp1.png'>

<footer style='size:0.1px;' > http://neuralnetworksanddeeplearning.com/</footer>


--- .class #id

## Basic Concepts

* Activation functions and Cost Functions

<img src = 'mlp2.png'>

* For example : Excel Demo


--- .class #id

## Gradient Descent: Optimising a convex function

* We have seen that there is always a cost function, to minimize cost, Gradient Descent and its variations are used.

* The primary identity is:
 
 $$w^{new}=w^{old}-\eta \nabla C(w)$$ 
* Here $w^{new}, w^{old}$ are model parameter vectors, $C(w)$ is the cost function and $\nabla$ is the gradient


--- .class #id
 

## Gradient Descent: Optimising a convex function

* This is a simplistic representation:

<img src= 'gd.png'>


--- .class #id

## Backpropagation:

* Different weight vectors in different layers

<img src='backprop.gif'>

* A more involved (one of the best discussions I could find) can be accessed from here http://neuralnetworksanddeeplearning.com/chap2.html


--- .class #id


## Convolutional Neural Networks

* Flattening an image and using it as an input has some disadvantages:
 * There is information in spatial arrangement, flattening might destroy that
 * The number of weights to be estimated will increase if neural networks with many layers and moderate resolution is used. (224*224 pixels and RGB channels)

* CNN reduce the number of weights to be estimated and also preserve the spatial information


--- .class #id

## Convolutional Neural  Networks

* What is convolution?

<img src='conv1.gif'>

* Kernel

<img src='kernel.png'>

* Source: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html



--- .class #id

## Convolutional Neural Networks

* Related terms: Padding

<img src='conv2.gif'>

* A couple of factors affect the output size of output from a convolving kernel:
 * Zero padding
 * Kernel Size
 * Strides
* Source: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html 


--- .class #id

## Convolutional Neural Networks

* Effect of no padding, unit strides:

<img src='conv3.gif'>

* Source: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html


--- .class #id

## Convolutional Neural Networks
* Double padding, unit strides:

<img src='conv4.gif'>



--- .class #id

## Convolutional Neural Networks

* Same size output is also possible (is desirable)

<img src='conv5.gif'>

* Source: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html

--- .class #id

## Convolutional Neural Networks

* Neural Network with convolutional layers

<img src='depthcol.jpeg'>

* Assume there is a 5 by 5 kernel with a bias term and there are 11 such kernels in this layer, the how many parameters will be needed? Compare with a vanilla MLP?

* Source: http://cs231n.github.io/convolutional-networks/


--- .class #id

## Convolutional Neural Networks

* CNNs not only have convolutional layers, they also have pooling layers, that reduce the size of image

<img src='pool.jpeg'>

* Source: http://cs231n.github.io/convolutional-networks/


--- .class #id

## Convolutional Neural Networks
* This is how pooling layer works:

<img src='maxpool.jpeg'>

* Source: http://cs231n.github.io/convolutional-networks/


--- .class #id

## Convolutional Neural Networks

* Sample complete CNN architecture (LeNet5)

<img src='lenet5.png'>


--- .class #id

## Convolutional Neural Networks

* There are other popular architectures AlexNet, VGGNet, ResNet
* Even after using convolving kernels, CNNs are resource intensive and require large amounts of images to train.
* To tackle this issue, one can use a pre-trained model and fine tune it to specific classification tasks, this is called **Transfer Learning**


--- .class #id

## Convolutional Neural Networks: Transfer Learning
 
* Here is a schematic:
 
 <img src='transfer_learning.jpg'>


--- .class #id
