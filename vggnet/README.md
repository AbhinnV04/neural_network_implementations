
# VGGNet Implementation

Welcome to the VGGNet implementation! This repository contains the code for implementing VGGNet from scratch and using transfer learning with a pre-trained VGG-16 model. The goal is to provide a clear and modular implementation for educational purposes.

## VGGNet Architecture

VGGNet is a convolutional neural network architecture known for its simplicity and effectiveness. The implementation includes both a standalone version created from scratch and a transfer learning version using a pre-trained VGG-16 model.

![fig 1.0](https://media.geeksforgeeks.org/wp-content/uploads/20200219152207/new41.jpg)

### Code Overview

The implementation is organized into two main parts:

1. **Standalone Implementation \:**
    - Located in `vggnet.py`.
    - Defines functions to add convolutional and dense blocks.
    - Creates the VGG-16 model using these blocks with a specified input shape.
    - Provides a summary of the model architecture.

```python
def add_conv_block(model, filters, num_layers=2):
    for _ in range(num_layers):
        model.add(Conv2D(filters, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, strides=2))
    return model 

def add_dense_block(model, units, activation, dropout_rate=None):
    model.add(Dense(units, activation=activation))
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    return model

def vgg_16_model(input_shape=(200, 200, 3)):
    model = Sequential()
    # Convolutional Layers
    model.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, strides=2))
    
    model = add_conv_block(model, 128)
    model = add_conv_block(model, 256)
    model = add_conv_block(model, 512)
    model = add_conv_block(model, 512)

    model.add(Flatten())
    # Fully Connected Layers
    model = add_dense_block(model, 4096, 'relu')
    model = add_dense_block(model, 4096, 'relu')
    model = add_dense_block(model, 1, 'sigmoid')

    return model

model = vgg_16_model()
model.summary()
```

   | Layer (type)               | Output Shape            | Param #    |
|----------------------------|-------------------------|------------|
| conv2d (Conv2D)             | (None, 200, 200, 64)    | 1,792      |
| conv2d_1 (Conv2D)           | (None, 200, 200, 64)    | 36,928     |
| max_pooling2d (MaxPooling2D)| (None, 100, 100, 64)    | 0          |
| conv2d_2 (Conv2D)           | (None, 100, 100, 128)   | 73,856     |
| conv2d_3 (Conv2D)           | (None, 100, 100, 128)   | 147,584    |
| max_pooling2d_1 (MaxPooling2D)| (None, 50, 50, 128)    | 0          |
| conv2d_4 (Conv2D)           | (None, 50, 50, 256)     | 295,168    |
| conv2d_5 (Conv2D)           | (None, 50, 50, 256)     | 590,080    |
| max_pooling2d_2 (MaxPooling2D)| (None, 25, 25, 256)    | 0          |
| conv2d_6 (Conv2D)           | (None, 25, 25, 512)    | 1,180,160  |
| conv2d_7 (Conv2D)           | (None, 25, 25, 512)    | 2,359,808  |
| max_pooling2d_3 (MaxPooling2D)| (None, 12, 12, 512)    | 0          |
| conv2d_8 (Conv2D)           | (None, 12, 12, 512)    | 2,359,808  |
| conv2d_9 (Conv2D)           | (None, 12, 12, 512)    | 2,359,808  |
| max_pooling2d_4 (MaxPooling2D)| (None, 6, 6, 512)      | 0          |
| flatten (Flatten)           | (None, 18432)           | 0          |
| dense (Dense)               | (None, 4096)            | 75,501,568 |
| dense_1 (Dense)             | (None, 4096)            | 16,781,312 |
| dense_2 (Dense)             | (None, 1)               | 4,097      |
| Total params:               |                         | 101,691,969|
| Trainable params:           |                         | 101,691,969|
| Non-trainable params:       |                         | 0          |


2. **Transfer Learning Implementation :**
    - Uses transfer learning with a pre-trained VGG-16 model from Keras.
    - Freezes the weights of the pre-trained layers and adds custom fully connected layers for a specific task (binary classification).
    - Compiles the model and provides a summary.

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

for layer in base_model.layers:
    layer.trainable = False

model_2 = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  
])

```

| Layer (type)               | Output Shape            | Param #    |
|----------------------------|-------------------------|------------|
| vgg16 (Functional)         | (None, 6, 6, 512)       | 14,714,688 |
| flatten_1 (Flatten)        | (None, 18432)           | 0          |
| dense_3 (Dense)            | (None, 512)             | 9,437,696  |
| dense_4 (Dense)            | (None, 1)               | 513        |
| Total params:               |                         | 24,152,897 |
| Trainable params:           |                         | 9,438,209  |
| Non-trainable params:       |                         | 14,714,688 |

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/vggnet-implementation.git
   cd vggnet-implementation
   ```

2. **Explore the Implementations:**
   - Navigate to the standalone and transfer learning directories to find the implementation details.
   - Understand the code structure and architecture specifics.

3. **Run the Code:**
   - Follow the instructions in each directory's README or code comments to run the implementations.

## Acknowledgments

- This implementation is inspired by the original VGGNet paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
