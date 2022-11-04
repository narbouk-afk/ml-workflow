from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
import numpy as np

class AbstractModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, kf : KFold, X: np.array, y: np.array):
        pass

    def predict(self, X_test) -> np.array:
        pass

    def getModel(self):
        return self.model

    @abstractmethod
    def metric(self):
        pass

class SklearnModel(AbstractModel):

    def __init__(self, model):
        super().__init__(model)

    def train(self, kf: KFold, X: np.array, y: np.array):
        precisions, recalls = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            precision, recall = compute_precision_recall(y_test, y_pred)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls

    def get_model(self):
        return self.model

    def predict(self, X_test) -> np.array:
        return self.model.predict(X_test)
def compute_precision_recall( y_pred, y_true):
    temp_recall = y_pred[y_true == 1]
    temp_precision = y_true[y_pred == 1]
    recall = np.sum(temp_recall == 1) / temp_recall.size
    precision = np.sum(temp_precision == 1) / temp_precision.size
    return precision, recall