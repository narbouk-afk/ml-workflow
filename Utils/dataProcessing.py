import numpy as np
import pandas as pd
import os
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def get_dataset_name(path: str) -> str:
    temp = path.replace("\\", "/")
    temp = temp.split("/")
    return temp[-1]


def import_data(path: str, header='infer') -> pd.DataFrame:
    data = pd.read_csv(f'{os.getcwd()}/{path if path[0] != "/" else path[1:]}', sep=",", header=header)
    return data


def clean_data(data: pd.DataFrame,
               corrections=None,
               categorical_columns=None,
               mode='mean',
               n_components=2) -> (NDArray[np.float], NDArray[np.int]):

    if categorical_columns is None:
        categorical_columns = []
    if corrections is None:
        corrections = []

    for correction in corrections:
        data = data.replace(correction[0], correction[1])
    data = data.replace("?", np.nan)

    # Define Imputer to replace missing values
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp = SimpleImputer(missing_values=np.nan, strategy=mode)

    if len(categorical_columns) > 0:
        imp_most_frequent.fit(data[categorical_columns])
        # Replace categorical missing value with most frequent value
        data[categorical_columns] = imp_most_frequent.transform(
            data[categorical_columns])
        data[categorical_columns] = data[categorical_columns].astype("category")

    numeric_columns = data.columns[data.dtypes != "category"]
    # Convert non categorical column into numeric
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
    imp.fit(data[numeric_columns])
    # Replace numeric missing value with column's mean or median
    data[numeric_columns] = imp.transform(data[numeric_columns])

    y = data.iloc[:, -1].values
    y = y.reshape(-1)

    data = pd.get_dummies(data)  # one hot encode
    data = (data - data.mean()) / data.std()  # normalize data

    X = data.iloc[:,:-1].values

    # Dimension reduction using PCA
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)

    return X, y
