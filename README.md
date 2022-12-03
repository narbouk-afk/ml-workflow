# ML Project



## Installation

For installation, the easiest way is to clone this repository using 
```
git clone https://gitlab.imt-atlantique.fr/y20xu/ml-project.git
```

You can also download it as an archive file.

## Dependencies

Make sure to have the following dependencies installed : 
```
- numpy 
- pandas
- sklearn
- pytorch
```

## How to run the script

To run the project, you need to specify the path where the dataset is stored and a model name :
```
python3 main.py dataset/path model_name
```
Different models have been implemented, you can choose a model in the following list :
```
       Model        |   model_name   | Library used
--------------------+----------------+--------------
        SVM         |      SVC       |   Sklearn
--------------------+----------------+--------------
        MLP         |    TorchMLP    |   PyTorch
--------------------+----------------+--------------
Logistic regression | LogRegression  |   Sklearn
--------------------+----------------+--------------
   Decision tree    |  DecisionTree  |   Sklearn
--------------------+----------------+--------------
   Random forest    |  RandomForest  |   Sklearn
--------------------+----------------+--------------
        KNN         |      KNN       |   Sklearn
--------------------+----------------+--------------
    Naive Bayes     |   NaiveBayes   |   Sklearn
```
## About models' parameters/architecture

All Sklearn models have been configured with default parameters. To change models' parameters, go to `params.py`
and add the parameters you want to modify in the `arg` dictionary associated with the model 
you want to modify.

TorchMLP is a basic MLP implemented with Pytorch.
It is a fully connected Neural Network with 3 hidden layers :
```
Hidden Layer 1 | 64 Neurals | Dropout : False | Activation : ReLU 
Hidden Layer 2 | 32 Neurals | Dropout : True  | Activation : ReLU 
Hidden Layer 3 | 16 Neurals | Dropout : True  | Activation : ReLU 
Output Layer   |  1 Neural  | Dropout : True  | Activation : Sigmoid
```

## Test

To run tests, just run the following command in the root folder :
```
pytest
```
Make sure you have `pytest` already installed

