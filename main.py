import sys
from Utils import *

def main(data_path, model_name):
    data = import_data(data_path)
    X, y = clean_data(data)
    X_train, X_test, y_train, y_test = split_data(X,y,0.8)
    nb_input = X.shape()
    model = get_model(model_name, nb_input)
    kf = k_fold_cross_validation()
    precisions_train, recalls_train = model.train(X_train,y_train,kf)
    model.getModel().save() # à voir
    y_predict = model.predict(X_test)
    precisions_test, recall_test = compute_precision_recall(y_predict, y_test)
    print(precisions_train, recalls_train, precisions_test, recall_test)

if __name__ == '__main__':
    data_path = sys.argv[1]
    model_name = sys.argv[2]
    main(data_path, model_name)

