# AlexNet Model

## Introduction

![AlexNet](https://miro.medium.com/v2/resize:fit:1400/1*1I09VdzcvvZHr-BTG2IwYA.jpeg)

This repository contains the implementation of the AlexNet model, a groundbreaking convolutional neural network (CNN) architecture that significantly contributed to the advancement of computer vision tasks. AlexNet was introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, and it won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, marking a pivotal moment in the history of computer vision.

## Model Architecture

AlexNet consists of eight layers, including five convolutional layers and three fully connected layers. The architecture can be summarized as follows:
<br>
**TensorFlow Implementation :** 

```python
Sequential([
        Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(3, strides=2),
        Conv2D(256, 5, padding='same', activation='relu'),
        MaxPooling2D(3, strides=2),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(3, strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])
```

| Layer (type)               | Output Shape            | Param #    |
|----------------------------|-------------------------|------------|
| conv2d_5 (Conv2D)           | (None, 55, 55, 96)      | 34,944     |
| max_pooling2d_3 (MaxPooling2D)| (None, 27, 27, 96)     | 0          |
| conv2d_6 (Conv2D)           | (None, 27, 27, 256)     | 614,656    |
| max_pooling2d_4 (MaxPooling2D)| (None, 13, 13, 256)    | 0          |
| conv2d_7 (Conv2D)           | (None, 13, 13, 384)     | 885,120    |
| conv2d_8 (Conv2D)           | (None, 13, 13, 384)     | 1,327,488  |
| conv2d_9 (Conv2D)           | (None, 13, 13, 256)     | 884,992    |
| max_pooling2d_5 (MaxPooling2D)| (None, 6, 6, 256)      | 0          |
| flatten_1 (Flatten)         | (None, 9216)            | 0          |
| dense_3 (Dense)             | (None, 4096)            | 37,752,832 |
| dense_4 (Dense)             | (None, 4096)            | 16,781,312 |
| dense_5 (Dense)             | (None, 1000)            | 4,097,000  |
| Total params:               |                         | 62,378,344 |
| Trainable params:           |                         | 62,378,344 |
| Non-trainable params:       |                         | 0          |

***Disclaimer: Keras does not have an offical AlexNet Model available to be utilized for transfer Learning.***

## Relevance in the History of Computer Vision

AlexNet played a pivotal role in the history of computer vision, as it demonstrated the effectiveness of deep learning models in image classification tasks. Its victory in the ILSVRC 2012 competition marked the beginning of the deep learning era in computer vision. The novel architecture, parallelization techniques, and the utilization of Rectified Linear Units (ReLU) activation functions contributed to its success.

## Acknowledgements

The AlexNet model was introduced in the paper:
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, presented at the Neural Information Processing Systems (NeurIPS) conference in 2012.
