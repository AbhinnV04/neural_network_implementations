# LeNet-5 and Custom CNN Model for Image Classification

## Introduction

This repository contains implementations and analysis of the LeNet-5 architecture and a custom Convolutional Neural Network (CNN) model for image classification. The models are trained and evaluated on the CIFAR-10 dataset.

## LeNet-5 Architecture
![fig 1.0](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-18-12-47-59.png) ![fig 1.1](https://www.datasciencecentral.com/wp-content/uploads/2021/10/1gNzz6vvWmF6tDN6pTRTd9g.jpeg)

The original **LeNet-5** model, proposed by Yann LeCun and his colleagues, was designed for grayscale images of size (32, 32, 1) to tackle the MNIST dataset. The architecture involves convolutional layers, max-pooling, and fully connected layers. It played a pivotal role in the development of convolutional neural networks.

## Custom CNN Model

The custom CNN model, defined in the `cnn_model` function, is tailored for the CIFAR-10 dataset, which consists of 32x32 color images across 10 classes. The architecture is as follows:

```python
def cnn_model():
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

This model is designed to process 3-channel (RGB) images, and it employs max-pooling and the Adam optimizer.

## Model Training and Results
The model was trained for 10 epochs on the CIFAR-10 dataset, and the results indicate the following:

|Metric | Performance |
| ----------- | ----------------- |
|Test Loss| 1.9489763975143433|
|Test Accuracy| 0.5799999833106995|

## Analysis
The observed lower accuracy can be attributed to the model's simplicity and limited capacity to capture complex features in the CIFAR-10 dataset. More sophisticated architectures like VGG, ResNet, or DenseNet are often preferred for such tasks due to their ability to learn hierarchical representations.