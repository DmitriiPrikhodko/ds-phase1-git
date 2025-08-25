import numpy as np

y_mse = np.array([0, 0, 0, 0, 0])
y_acc = np.array([0, 1, 1, 1, 1])

def test_mse(pred, y=y_mse):
    return np.sum((pred - y)**2) / len(y)

def test_acc(pred, y=y_acc):
    return np.sum(pred == y_acc)/len(pred)

