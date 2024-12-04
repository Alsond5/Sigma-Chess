import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
import numpy as np

print("All devices: ", tf.config.list_logical_devices('GPU'))

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def residual_block(inputs, filters=256, kernel_size=(3, 3), stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, inputs])
    x = layers.ReLU()(x)
    
    return x

def sigmachess_network(input_shape=(8, 8, 119)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(256, (3, 3), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(19):
        x = residual_block(x)

    policy = layers.Conv2D(256, (3, 3), strides=1, padding="same")(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.ReLU()(policy)
    policy = layers.Conv2D(73, (1, 1), strides=1, padding="same")(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(4672, name="policy_output", activation="softmax")(policy)

    value = layers.Conv2D(1, (1, 1), strides=1, padding="same")(x)
    value = layers.BatchNormalization()(value)
    value = layers.ReLU()(value)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation="relu")(value)
    value = layers.Dense(1, activation="tanh", name="value_output")(value)

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