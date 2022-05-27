from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D


def build_model(
    height: int,
    width: int,
    channels: int,
    actions: int
) -> Sequential:
    model = Sequential()
    model.add(Convolution2D(
        filters=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        activation='relu',
        input_shape=(3, height, width, channels)
    ))
    model.add(Convolution2D(
        filters=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        activation='relu'
    ))
    model.add(Convolution2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(
        units=256,
        activation='relu'
    ))
    model.add(Dense(
        units=128,
        activation='relu'
    ))
    model.add(Dense(
        units=actions,
        activation='softmax'
    ))
    return model
