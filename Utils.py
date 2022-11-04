from sklearn.model_selection import KFold, train_test_split
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

    def get_model(self):
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

def import_data():
    return

def clean_data(X, y): # (X: np.array, y: np.array)
    return # np.array, np.array

def split_data(X, y, test_size): # (X: np.array, y: np.array, test_size :float)
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_model_by_name(name, nb_input):# -> AbstractModel
    # Match name:
    # Case “SVC”:
    #     Return SklearnModel(SVC())
    # Case “TorchMLP”:
    #     return TorchModel(MLP(nb_input))
    return

def k_fold_cross_validation(n_splits = 5, random_state=34):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kf

def compute_precision_recall( y_pred, y_true):
    temp_recall = y_pred[y_true == 1]
    temp_precision = y_true[y_pred == 1]
    recall = np.sum(temp_recall == 1) / temp_recall.size
    precision = np.sum(temp_precision == 1) / temp_precision.size
    return precision, recall


