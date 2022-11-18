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

## Test and Deploy
To run tests, just run the following command in the root folder :
```
pytest
```
Make sure you have `pytest` already installed

