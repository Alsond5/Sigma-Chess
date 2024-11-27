import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def sigmachess_network(input_shape=(8, 8, 119), num_actions=4672):
    inputs = layers.Input(input_shape)

    x = layers.Conv2D(256, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(19):
        shortcut = x

        x = layers.Conv2D(256, (3, 3), padding="same")(shortcut)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(256, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)

    policy_output = layers.Dense(num_actions, activation="softmax", name="policy_output")(x)
    value_output = layers.Dense(1, activation="sigmoid", name="value_output")(x)

    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(optimizer=Adam(learning_rate=0.001), loss={
        "policy_output": "categorical_crossentropy",
        "value_output": "mean_squared_error"
    },
    metrics={
        "policy_output": "accuracy",
        "value_output": "mse"
    })

    return model