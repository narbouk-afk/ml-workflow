from sklearn.model_selection import KFold


def clean_data(X, y): # (X: np.array, y: np.array)
    return # np.array, np.array
def split_data(X, y, c): # (X: np.array, y: np.array, c:float)
    return
def compute_precision_recall(y_pred, y_true):
    precisions, recalls = 0,0
    return precisions, recalls
def k_fold_cross_validation(n_splits = 5, random_state=34):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=34)
    return kf
def import_data():
    return
def get_model(name):
    return #