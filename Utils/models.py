from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch
import sys


class AbstractModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, X: np.array, y: np.array, kf: KFold):
        pass

    @abstractmethod
    def predict(self, X_test) -> np.array:
        pass

    def get_model(self):
        return self.model


class SklearnModel(AbstractModel):
    def __init__(self, model):
        super().__init__(model)

    def train(self, X: np.array, y: np.array, kf: KFold):
        precisions, recalls = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_valid)
            precision, recall = compute_precision_recall(y_valid, y_pred)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls

    def get_model(self):
        return self.model

    def predict(self, X_valid) -> np.array:
        return self.model.predict(X_valid)


class TorchModel(AbstractModel):
    def __init__(self, model: nn.Module, n_epochs=50, lr=0.01):
        super().__init__(model)
        self.fold_index = 0
        self.n_split = 0
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.device = torch.device('cpu')
        self.model.to(device=self.device)  # use cpu for Neural Network

    def train(self, X: np.array, y: np.array, kf: KFold) -> (list[float], list[float]):
        precisions, recalls = [], []
        self.fold_index = 0
        self.n_split = kf.n_splits
        self.__progress()
        for train_index, test_index in kf.split(X):
            X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]

            # Create DataLoader for training
            tensor_X_train = torch.from_numpy(X_train)
            tensor_y_train = torch.from_numpy(y_train)

            train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
            train_loader = DataLoader(train_dataset)

            # Train the model on one split
            self.__one_folded_train(train_loader)

            # Validate the model
            y_pred = self.predict(X_valid)
            precision, recall = compute_precision_recall(y_pred, y_valid)
            precisions.append(precision)
            recalls.append(recall)
            self.fold_index += 1
            self.__progress()
        print("")
        return precisions, recalls

    def __one_folded_train(self, train_loader):
        for epoch in range(self.n_epochs):
            train_loss, valid_loss = 0, 0  # monitor losses

            # train the model
            self.model.train()  # prep model for training
            for data, label in train_loader:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32).unsqueeze(1)
                self.optimizer.zero_grad()  # clear the gradients of all optimized variables
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                loss = self.criterion(output, label)  # calculate the loss
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                self.optimizer.step()  # perform a single optimization step (parameter update)
                train_loss += loss.item() * data.size(0)  # update running training loss
        pass

    def predict(self, X_test) -> np.array:
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(X_test)
                                .float()
                                .to(self.device))
        return torch.round(output).to('cpu').numpy().reshape(-1)

    def __progress(self):
        bar_len = 100
        filled_len = int(round(bar_len * self.fold_index / float(self.n_split)))

        percents = round(100.0 * self.fold_index / float(self.n_split), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s \r' % (bar, percents, '%'))
        sys.stdout.flush()


class TorchMLP(nn.Module):
    def __init__(self, nb_input):
        super(TorchMLP, self).__init__()
        self.fc1 = nn.Linear(nb_input, 64)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigm(self.fc4(x))
        return x


def get_model_by_name(name: str, nb_input: int, **kwargs) -> AbstractModel:  # -> AbstractModel
    match name:
        case "SVC":
            return SklearnModel(SVC(**kwargs))
        case "TorchMLP":
            return TorchModel(TorchMLP(nb_input))
        case "LogRegression":
            return SklearnModel(LogisticRegression(**kwargs))
        case "DecisionTree":
            return SklearnModel(DecisionTreeClassifier(**kwargs))
        case "RandomForest":
            return SklearnModel(RandomForestClassifier(**kwargs))
        case "KNN":
            return SklearnModel(KNeighborsClassifier(**kwargs))
        case "NaiveBayes":
            return SklearnModel(GaussianNB(**kwargs))
        case _:
            raise NameError('No matching model found')


def compute_precision_recall(y_pred, y_true):
    temp_recall = y_pred[y_true == 1]  # retrieve predicted class of real true class
    temp_precision = y_true[y_pred == 1]  # retrieve real class of predicted true class
    recall = np.sum(temp_recall == 1) / temp_recall.size
    precision = np.sum(temp_precision == 1) / temp_precision.size
    return precision, recall
