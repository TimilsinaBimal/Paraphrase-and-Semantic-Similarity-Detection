from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow import keras
from utils.layer_utils import calculate_mean, normalize


def binary_base_model_GRU(vocab):
    base_model = models.Sequential()
    base_model.add(layers.Embedding(input_dim=len(vocab) + 2, output_dim=128))
    base_model.add(layers.Bidirectional(
        layers.GRU(128, return_sequences=True)))
    base_model.add(layers.Bidirectional(
        layers.GRU(128, return_sequences=True)))
    base_model.add(layers.GlobalAveragePooling1D())
    base_model.add(layers.Dense(128, activation='relu'))
    base_model.add(layers.Dropout(0.3))
    base_model.add(layers.Dense(128, activation='relu'))
    return base_model


def binary_base_model_LSTM(vocab):
    base_model = models.Sequential()
    base_model.add(layers.Embedding(input_dim=len(vocab) + 2, output_dim=128))
    base_model.add(layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)))
    base_model.add(layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)))
    base_model.add(layers.GlobalAveragePooling1D())
    base_model.add(layers.Dense(128, activation='relu'))
    base_model.add(layers.Dropout(0.3))
    base_model.add(layers.Dense(128, activation='relu'))
    return base_model


def binary_base_cnn_model(vocab):
    base_model = models.Sequential()
    base_model.add(layers.Embedding(input_dim=len(vocab) + 2, output_dim=128))
    base_model.add(layers.Conv1D(128, 3, padding='same'))
    base_model.add(layers.MaxPool1D())
    base_model.add(layers.BatchNormalization())
    base_model.add(layers.Dropout(0.2))
    base_model.add(layers.Conv1D(128, 3))
    base_model.add(layers.MaxPool1D())
    base_model.add(layers.Dropout(0.3))
    base_model.add(layers.Dense(128, activation='relu'))
    base_model.add(layers.Dropout(0.2))
    base_model.add(layers.Dense(128, activation='relu'))
    return base_model


def binary_model(vocab, data):
    input1 = keras.Input(shape=(data.shape[1],))
    input2 = keras.Input(shape=(data.shape[1],))

    base_model = binary_base_model_GRU(vocab)

    encoding1 = base_model(input1)
    encoding2 = base_model(input2)

    distance = layers.Concatenate()([encoding1, encoding2])
    dense1 = layers.Dense(128, activation='relu')(distance)
    bn = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(56, activation='relu')(bn)
    final = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input1, input2], outputs=final)
    return model


def binary_model_GRU(vocab, data):
    input1 = keras.Input(shape=(data.shape[1],))
    input2 = keras.Input(shape=(data.shape[1],))

    base_model = binary_base_model_GRU(vocab)

    encoding1 = base_model(input1)
    encoding2 = base_model(input2)

    distance = layers.Concatenate()([encoding1, encoding2])
    dense1 = layers.Dense(128, activation='relu')(distance)
    bn = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(128, activation='relu')(bn)
    final = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input1, input2], outputs=final)
    return model


def binary_model_LSTM(vocab, data):
    input1 = keras.Input(shape=(data.shape[1],))
    input2 = keras.Input(shape=(data.shape[1],))

    base_model = binary_base_model_LSTM(vocab)

    encoding1 = base_model(input1)
    encoding2 = base_model(input2)

    distance = layers.Concatenate()([encoding1, encoding2])
    dense1 = layers.Dense(128, activation='relu')(distance)
    bn = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(128, activation='relu')(bn)
    final = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input1, input2], outputs=final)
    return model


def binary_model_CNN(vocab, data):
    input1 = keras.Input(shape=(data.shape[1],))
    input2 = keras.Input(shape=(data.shape[1],))

    base_model = binary_base_cnn_model(vocab)

    encoding1 = base_model(input1)
    encoding2 = base_model(input2)

    distance = layers.Concatenate()([encoding1, encoding2])
    dense1 = layers.Dense(128, activation='relu')(distance)
    bn = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(128, activation='relu')(bn)
    final = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input1, input2], outputs=final)
    return model


def triplet_base_model(vocab):
    base_model = tf.keras.Sequential()
    base_model.add(tf.keras.layers.Embedding(
        input_dim=len(vocab) + 2, output_dim=128))
    base_model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    base_model.add(tf.keras.layers.Lambda(calculate_mean, name='mean'))
    base_model.add(tf.keras.layers.Lambda(normalize, name='normalize'))
    return base_model


def triplet_model(vocab, data):
    input1 = tf.keras.layers.Input(shape=(data.shape[1],))
    input2 = tf.keras.layers.Input(shape=(data.shape[1],))

    base_model = triplet_base_model(vocab)

    encoding1 = base_model(input1)
    encoding2 = base_model(input2)

    merged = tf.keras.layers.Concatenate()([encoding1, encoding2])

    model = tf.keras.Model(inputs=[input1, input2], outputs=merged)
    return model
