import numpy as np

class AbstractAdapter:
    def __init__(self, model):
        self.model = model
    def train(self, X_train:np.array, y_train: np.array):
        return 
    def predict(self, X_test):
        return #numpy array
    def getModel(self):
        return #model
    def metric(self):
        return
