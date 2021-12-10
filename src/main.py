import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from data.preprocessing import prepare_data, read_data
from models.predict import predict_binary, predict_binary_single
from utils.config import (
    BATCH_SIZE, BINARY_VOCAB_PATH, DEV_DATA_PATH,
    EPOCHS, TEST_DATA_PATH, TRAIN_DATA_PATH,
    LEARNING_RATE
)

# df_train = read_data(TRAIN_DATA_PATH)
# df_dev = read_data(DEV_DATA_PATH)
df_test = read_data(TEST_DATA_PATH)


# df_train_pos = df_train.loc[(df_train['similarity'] == 1)]
# df_dev_pos = df_dev.loc[(df_dev['similarity'] == 1)]
# df_train_pos.drop_duplicates(inplace=True)
# df_dev_pos.drop_duplicates(inplace=True)
# train_sen1 = df_train['sentence1'].tolist()
# train_sen2 = df_train['sentence2'].tolist()

# dev_sen1 = df_dev['sentence1'].tolist()
# dev_sen2 = df_dev['sentence2'].tolist()

test_sen1 = df_test['sentence1'].tolist()
test_sen2 = df_test['sentence2'].tolist()

# train_label = df_train['similarity'].tolist()
# dev_label = df_dev['similarity'].tolist()
test_label = df_test['similarity'].tolist()


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
    # sentence1 = input('Enter Sentence 1: ')
    # sentence2 = input('Enter Sentence 2: ')
    data1, data2 = prepare_data(
        test_sen1,
        test_sen2,
        BINARY_VOCAB_PATH,
        training=False
    )
    model = models.load_model('binary_loss_model1.h5')
    preds = predict_binary(model, [data1, data2])
    print(preds)
    print(np.sum(np.array(preds) == np.array(test_label)) / len(test_label))


make_predictions()
