from tensorflow.keras import backend as K


def calculate_mean(x, axis=1):
    return K.mean(x, axis=axis)


def normalize(x):
    return x / K.sqrt(K.sum(x * x, axis=-1, keepdims=True))
