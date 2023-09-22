import numpy as np

def create_confusion_matrix(actual, predictions):
    cm = np.zeros((10, 10), dtype=int)
    for a, p in zip(actual, predictions):
        cm[a][p] += 1

    return cm

