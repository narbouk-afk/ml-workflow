import sys
from sklearn.model_selection import train_test_split
from Utils.models import *
from Utils.dataProcessing import *
from Utils.params import data_params, model_params
from Utils.render import display_result


def main(data_path: str, model_name: str):
    # retrieve some parameters
    d_params = data_params[get_dataset_name(data_path)]
    m_params = model_params[model_name]

    # import data and preprocessing
    data = import_data(data_path)
    X, y = clean_data(data, **d_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model and training
    nb_input = X.shape[1]
    model = get_model_by_name(model_name, nb_input, **m_params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    precisions_train, recalls_train = model.train(X_train, y_train, kf)

    # Test the model
    y_predict = model.predict(X_test)
    precisions_test, recall_test = compute_precision_recall(y_predict, y_test)
    display_result(precisions_train, recalls_train, precisions_test, recall_test)


if __name__ == '__main__':
    data_path = sys.argv[1]
    model_name = sys.argv[2]
    main(data_path, model_name)
