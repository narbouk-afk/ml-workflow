import numpy as np
import pandas as pd
import os
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer


def get_dataset_name(path: str) -> str:
    temp = path.replace("\\", "/")
    temp = temp.split("/")
    return temp[-1]


def import_data(path: str, header='infer') -> pd.DataFrame:
    print(path)
    data = pd.read_csv(f'{os.getcwd()}/{path if path[0] != "/" else path[1:]}', sep=",", header=header)
    return data


def clean_data(data: pd.DataFrame,
               corrections=None,
               categorical_columns=None,
               mode='mean') -> (NDArray[np.float], NDArray[np.int]):
    if categorical_columns is None:
        categorical_columns = []
    if corrections is None:
        corrections = []
    for correction in corrections:
        data = data.replace(correction[0], correction[1])
    data = data.replace("?", np.nan)

    # Replace categorical missing value with most frequent value
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # Replace numeric missing value with column's mean or median
    imp = SimpleImputer(missing_values=np.nan, strategy=mode)

    if len(categorical_columns) > 0:
        data[categorical_columns] = data[categorical_columns].astype("category")
        imp_most_frequent.fit(data[categorical_columns])
        data[categorical_columns] = imp_most_frequent.transform(
            data[categorical_columns])

    numeric_columns = data.columns[data.dtypes != "category"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
    imp.fit(data[numeric_columns])
    data[numeric_columns] = imp.transform(data[numeric_columns])

    y = data.iloc[:, -1].values
    y = y.reshape(-1)

    data = pd.get_dummies(data)  # one hot encode
    data = (data - data.mean()) / data.std()  # normalize data
    X = data.iloc[:, :-1].values

    return X, y  # X: np.array, y: np.array
