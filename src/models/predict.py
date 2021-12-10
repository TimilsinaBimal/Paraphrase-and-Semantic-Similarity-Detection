import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def predict_binary(model, data):
    test_data1, test_data2 = data
    preds = model.predict([test_data1, test_data2])
    preds_list = [1 if p[0] >= 0.5 else 0 for p in preds]
    return preds_list


def predict_binary_single(model, sentence1, sentence2):
    y_pred = model.predict([sentence1, sentence2])
    return 1 if y_pred[0] >= 0.5 else 0


def draw_confusion_matrix(y_test, y_pred, labels):
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.4)
    df_cm = pd.DataFrame(conf_mat, index=labels, columns=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, cmap=plt.get_cmap('OrRd'), fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
