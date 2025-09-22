import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier


def logistic_regression_model(C=1.0, max_iter=5000, n_jobs=-1):

    base_clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",  # "saga" works too for larger data
        n_jobs=n_jobs
    )
    model = OneVsOneClassifier(base_clf, n_jobs=n_jobs)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)