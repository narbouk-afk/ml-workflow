from abc import ABC, abstractmethod
import numpy as np

class AbstractModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, X_train: np.array, y_train: np.array) -> None:
        pass

    def predict(self, X_test) -> np.array:
        pass

    def getModel(self):
        return self.model
    @abstractmethod
    def metric(self):
        pass
