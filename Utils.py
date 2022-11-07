from sklearn.model_selection import KFold, train_test_split
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import torch.nn as nn
import torch

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

class SklearnModel(AbstractModel):
    def __init__(self, model):
        super().__init__(model)

    def train(self, kf: KFold, X: np.array, y: np.array):
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
    def __init__(self, model: nn.Module, n_epochs=30, lr=0.01):
        super().__init__(model)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        self.n_epochs = n_epochs
        self.init_device()


    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def train(self, kf: KFold, X: np.array, y: np.array) -> (list[float], list[float]):
        precisions, recalls = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]

            tensor_X_train, tensor_X_valid = torch.from_numpy(X_train), torch.from_numpy(X_valid)
            tensor_y_train, tensor_y_test = torch.from_numpy(y_train), torch.from_numpy(y_valid)

            train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
            valid_dataset = TensorDataset(tensor_X_valid, tensor_y_test)

            train_loader = DataLoader(train_dataset)
            valid_loader = DataLoader(valid_dataset)

            self.__one_folded_train(train_loader, valid_loader)

            y_pred = self.predict(X_valid)
            precision, recall = compute_precision_recall(y_pred, y_valid)
            precisions.append(precision)
            recalls.append(recall)
            return precisions, recalls

    def __one_folded_train(self, train_loader, valid_loader):
        valid_loss_min = np.Inf  # set initial "min" to infinity
        for epoch in range(self.n_epochs):
            train_loss, valid_loss = 0, 0  # monitor losses

            # train the model
            self.model.train()  # prep model for training
            for data, label in train_loader:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)
                self.optimizer.zero_grad()  # clear the gradients of all optimized variables
                output = self.model(data)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = self.criterion(output, label)  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step()  # perform a single optimization step (parameter update)
                train_loss += loss.item() * data.size(0)  # update running training loss

            # validate the model
            self.model.eval()
            for data, label in valid_loader:
                data = data.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.long)
                with torch.no_grad():
                    output = self.model(data)
                loss = self.criterion(output, label)
                valid_loss += loss.item() * data.size(0)

            # calculate average loss over an epoch
            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                torch.save(self.model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
        pass

    def predict(self, X_test) -> np.array:
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(X_test))
        return output


class TorchMLP(nn.Module):
    def __init__(self, nb_input): # FUNCTION TO BE COMPLETED
        super(TorchMLP,self).__init__()
        self.fc1 = nn.Linear(nb_input, 128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128,64)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64,32)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self,x): # FUNCTION TO BE COMPLETED
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigm(self.fc4(x))
        return x

def import_data(name='ckd', path):  # (name=’ckd’ / ‘bad’, path: str)
    if name == 'bad':
        data = pd.read_csv(path, sep=",", header=None)

    elif name == 'ckd':
        data = pd.read_csv(path)
        data.drop('id', axis=1, inplace=True)
        data.classification = data.classification.replace("ckd\t", "ckd")
        data.classification = data.classification.replace(
            ['ckd', 'notckd'], [1, 0])

    return data  # pd.DF


def clean_data(data, mode='means'):  # (data: pd.DF, mode = 'means' / 'median')
    imp_most_frequent = SimpleImputer(
        missing_values=np.nan, strategy='most_frequent')

    if mode == 'median':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    numeric_columns = data.columns[data.dtypes == float]
    categoric_columns = data.columns[data.dtypes ==
                                     object or data.dtypes == int]

    imp.fit(data[numeric_columns])
    imp_most_frequent.fit(data[categoric_columns])

    data[numeric_columns] = imp.transform(data[numeric_columns])
    data[categoric_columns] = imp_most_frequent.transform(
        data[categoric_columns])

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y

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


