import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.utils.vis_utils import plot_model

batch_size = 2

model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(256, 256, 3)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(15),
    ]
)
#Visualization I can't get to work on my pc
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

train = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=22,
    validation_split=0.1,
    subset="training",
)
validation = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=22,
    validation_split=0.1,
    subset="validation",
)


def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


train = train.map(augment)


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=["accuracy"],
)

model.fit(train, epochs=15, verbose=2)