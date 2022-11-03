import sys

def main(data, model):
    X, y = import_data(data)
    X_train, X_test, y_train, y_test = split_data(X,y,0.8)
    Model = Adapter(model())
    Model.train(X_train,y_train)
    y_predict = Model.predict(X_test)
if __name__ == '__main__':
    fileName = sys.argv[1]
    model = sys.argv[2]
    main(fileName, model)
