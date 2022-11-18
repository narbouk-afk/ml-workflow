data_params = {
    "kidney_disease.csv" : {
        "corrections": [("\t43", 43), ("\t6200", 6200), ("\t8400", 4800), ("\tno", "no"), ("\tyes", "yes"),
                         (" yes", "yes"), ("ckd\t", 1), ("\t?", ""), ("ckd", 1), ("notckd", 0)],
        "categoric_columns": ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    },
    "data_banknote_authentication.txt" : {
        "corrections" : [],
        "categoric_columns": []
    }
}

model_params = {
    "SVC": {
        "supervised": True,
        "args": {}
    },
    "TorchMLP": {
        "supervised": True,
        "args": {}
    },
    "LogRegression": {
        "supervised": True,
        "args": {}
    },
    "DecisionTree": {
        "supervised": True,
        "args": {}
    },
    "RandomForest": {
        "supervised": True,
        "args": {}
    },
    "KNN": {
        "supervised": False,
        "args": {}
    },
    "NaiveBayes": {
        "supervised": True,

    }
}