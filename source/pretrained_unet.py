import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import sys
import os
import random

def vgg16_unet(image_size=512, n_max_filters=512):

    # Load pretrained model
    vgg16 = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False)
    # Define input. It has 3 color channels since vgg is trained on a color dataset
    input = tf.keras.Input(shape=(image_size, image_size, 3))

    skip_connections = []

    # Get first conv block
    x = vgg16.layers[0](input)  # Input layer in vgg
    x = vgg16.layers[1](x)  # Conv layer
    x = vgg16.layers[2](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[3](x)  # Pooling layer

    # Get 2nd conv block
    x = vgg16.layers[4](x)  # Conv layer
    x = vgg16.layers[5](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[6](x)  # Pooling layer

    # Get 3rd conv block
    x = vgg16.layers[7](x)  # Conv layer
    x = vgg16.layers[8](x)  # Conv layer
    x = vgg16.layers[9](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[10](x)  # Pooling layer

    # Get 4th conv block
    x = vgg16.layers[11](x)  # Conv layer
    x = vgg16.layers[12](x)  # Conv layer
    x = vgg16.layers[13](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[14](x)  # Pooling layer

    # # Get 5th conv block
    x = vgg16.layers[15](x)  # Conv layer
    x = vgg16.layers[16](x)  # Conv layer
    x = vgg16.layers[17](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[18](x)  # Pooling layer

    # Starting upscaling and decoding
    for i, skip_i in enumerate(reversed(range(len(skip_connections)))):
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3), padding="same")(x) # Conv layer
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3), padding="same")(x) # Conv layer
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)),
                                            kernel_size=(3, 3), strides=2, padding="same")(x) # Upsample
        x = tf.concat([x, skip_connections[skip_i]], axis=-1) # Add skip connection to the channels

    # Last conv layers
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model



def run(image_size=512):
    model = vgg16_unet()


if __name__ == '__main__':
    run()






