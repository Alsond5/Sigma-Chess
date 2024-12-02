import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def residual_block(inputs, filters=256, kernel_size=(3, 3), stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", kernel_initializer=HeNormal())(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", kernel_initializer=HeNormal())(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, inputs])
    x = layers.ReLU()(x)
    
    return x

def sigmachess_network(input_shape=(8, 8, 119)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer=HeNormal())(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(19):
        x = residual_block(x)

    policy = layers.Conv2D(256, (3, 3), strides=1, padding="same", kernel_initializer=HeNormal())(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.ReLU()(policy)
    policy = layers.Conv2D(73, (1, 1), strides=1, padding="same", kernel_initializer=HeNormal())(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Softmax(name="policy_output")(policy)

    value = layers.Conv2D(1, (1, 1), strides=1, padding="same", kernel_initializer=HeNormal())(x)
    value = layers.BatchNormalization()(value)
    value = layers.ReLU()(value)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation="relu", kernel_initializer=HeNormal())(value)
    value = layers.Dense(1, activation="tanh", name="value_output", kernel_initializer=HeNormal())(value)

    model = models.Model(inputs=inputs, outputs=[policy, value])

    return model

def create_model():
    model = sigmachess_network()

    model.compile(
        optimizer=Adam(learning_rate=0.02),
        loss={
            "policy_output": "categorical_crossentropy",
            "value_output": "mean_squared_error"
        },
        metrics={
            "policy_output": "accuracy",
            "value_output": "mse"
        }
    )
    
    return model