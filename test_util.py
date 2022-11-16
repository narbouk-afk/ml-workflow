from Utils import compute_precision_recall
import numpy as np


def test_compute_precision_recall_1():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([1, 0, 1])
    precision, recall = compute_precision_recall(y_pred, y_true)
    assert ((precision == 1) and (recall == 1))

def test_compute_precision_recall_2():
    y_pred = np.array([0, 0, 1])
    y_true = np.array([1, 0, 1])
    precision, recall = compute_precision_recall(y_pred, y_true)
    assert ((precision == 1) and (recall == 0.5))


def test_compute_precision_recall_3():
    y_pred = np.array([1, 0, 1])
    y_true = np.array([0, 0, 1])
    precision, recall = compute_precision_recall(y_pred, y_true)
    assert ((precision == 0.5) and (recall == 1))


