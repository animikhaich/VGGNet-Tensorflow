import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

from .utils import conv3_unit, conv1_unit, maxpool_unit, flatten_unit, fc_unit, fc_final


def VGG13(shape=(224, 224, 3), classes=1000):
    """
    VGG13 Model Architecture

    VGGNet Architecture A Implementation
    13 Layers Deep - 10 Conv + 3 FC Layers

    Args:
        shape (tuple, optional): Input Image Shape. Defaults to (224, 224, 3).
        classes (int, optional): Number of Classes. Defaults to 1000.

    Returns:
        tf.Keras.models.Model: VGG11 Model
    """

    # Input Layer
    inputs = tf.keras.Input(shape=shape)

    # Block 1
    hidden = conv3_unit(64, inputs)
    hidden = conv3_unit(64, inputs)
    hidden = maxpool_unit(hidden)

    # Block 2
    hidden = conv3_unit(128, hidden)
    hidden = conv3_unit(128, hidden)
    hidden = maxpool_unit(hidden)

    # Block 3
    hidden = conv3_unit(256, hidden)
    hidden = conv3_unit(256, hidden)
    hidden = maxpool_unit(hidden)

    # Block 4
    hidden = conv3_unit(512, hidden)
    hidden = conv3_unit(512, hidden)
    hidden = maxpool_unit(hidden)

    # Block 5
    hidden = conv3_unit(512, hidden)
    hidden = conv3_unit(512, hidden)
    hidden = maxpool_unit(hidden)

    # FC Block
    hidden = flatten_unit(hidden)
    hidden = fc_unit(4096, hidden)
    hidden = fc_unit(4096, hidden)
    outputs = fc_final(classes, hidden)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
