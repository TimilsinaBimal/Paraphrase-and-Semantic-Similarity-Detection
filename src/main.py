from utils.config import (
    BATCH_SIZE, BINARY_VOCAB_PATH, DEV_DATA_PATH, EPOCHS,
    LEARNING_RATE, MODEL_PATH, TEST_DATA_PATH,
    TRAIN_DATA_PATH)
from models.predict import predict_binary, predict_binary_single
from data.preprocessing import prepare_data, read_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# df_train = read_data(TRAIN_DATA_PATH)
# df_dev = read_data(DEV_DATA_PATH)
# df_test = read_data(TEST_DATA_PATH)


# df_train_pos = df_train.loc[(df_train['similarity'] == 1)]
# df_dev_pos = df_dev.loc[(df_dev['similarity'] == 1)]
# df_train_pos.drop_duplicates(inplace=True)
# df_dev_pos.drop_duplicates(inplace=True)
# train_sen1 = df_train['sentence1'].tolist()
# train_sen2 = df_train['sentence2'].tolist()

# dev_sen1 = df_dev['sentence1'].tolist()
# dev_sen2 = df_dev['sentence2'].tolist()

# test_sen1 = df_test['sentence1'].tolist()
# test_sen2 = df_test['sentence2'].tolist()

# train_label = df_train['similarity'].tolist()
# dev_label = df_dev['similarity'].tolist()
# test_label = df_test['similarity'].tolist()


def compile_and_train(
        model, optimizer,
        loss, metrics: list,
        training_data: tuple,
        validation_data: tuple
):
    if metrics:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=loss
        )
    history = model.fit(
        training_data[0], training_data[1],
        batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=validation_data)
    return history.history


def make_predictions():
    # =============================
    # A woman with a green headscarf, blue shirt and a very big grin.
    # The woman is happy. <POSITIVE>
    # The woman is sad. <NEGATIVE>
    sentence1 = input('Enter Sentence 1: ')
    sentence2 = input('Enter Sentence 2: ')
    data1, data2 = prepare_data(
        [sentence1],
        [sentence2],
        BINARY_VOCAB_PATH,
        training=False
    )
    model = models.load_model(MODEL_PATH)
    preds = predict_binary_single(model, data1, data2)
    print("\n" * 3)
    print("===PARAPHRASE AND SEMANTIC SIMILARITY DETECTION===")
    print("SENTENCE 1: ", sentence1)
    print("SENTENCE 2: ", sentence2)
    print("===MODEL PREDICTION===")
    print("\n" * 1)
    if preds == 0:
        print("Sentence 1 and sentence 2 are semantically Different!")
    elif preds == 1:
        print("Sentence 1 and sentence 2 are semantically Similar!")

    print("\n" * 2)


make_predictions()
