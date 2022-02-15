import sklearn.metrics as sk
import numpy as np


def normalizeData(y_predict, y_true):
    y_predict = np.squeeze(y_predict)
    y_true = np.squeeze(y_true)
    return y_predict, y_true


def auc_roc(y_predict, y_true):
    y_predict, y_true = normalizeData(y_predict, y_true)
    auc_roc = sk.roc_auc_score(y_true, y_predict)
    return auc_roc


def recall(y_predict, y_true):
    y_predict, y_true = normalizeData(y_predict, y_true)
    ratio = 0.3
    num_outlier = int(ratio * y_true.shape[0])
    index_predict = np.argsort(-y_predict)
    prediction = np.zeros(y_true.size)
    prediction[index_predict[0:num_outlier]] = 1
    recall = sk.recall_score(y_true, prediction, average=None)
    return recall

