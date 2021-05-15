import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf


def conv3_unit(filters, inputs):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(inputs)


def conv1_unit(filters, inputs):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(inputs)


def maxpool_unit(inputs):
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)


def flatten_unit(inputs):
    return tf.keras.layers.Flatten()(inputs)


def fc_unit(units, inputs):
    fc_out = tf.keras.layers.Dense(units=units, activation="relu")(inputs)
    return tf.keras.layers.Dropout(rate=0.5)(fc_out)


def fc_final(classes, inputs):
    fc_out = tf.keras.layers.Dense(units=classes, activation=None)(inputs)
    return tf.keras.layers.Softmax(dtype=tf.float32)(fc_out)
