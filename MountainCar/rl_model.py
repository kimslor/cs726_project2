from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


def build_model(
    state_shape,
    actions: int
) -> Sequential:
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + state_shape))
    model.add(Dense(
        128,
        activation='relu'))
    model.add(Dense(
        64,
        activation='relu'))
    model.add(Dense(
        32,
        activation='relu'))
    model.add(Dense(
        actions,
        activation='linear'))
    model.summary()
    return model
