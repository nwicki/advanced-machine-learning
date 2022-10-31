import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RANSACRegressor, SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE, SelectPercentile
from scipy.stats import rankdata
import pandas
import numpy

def write_results(model):
    y_pred = model.predict(get_test_data())
    df = pandas.DataFrame(y_pred, columns=["y"])
    df.to_csv("submission.csv", index_label="id")

def test(X, y, model):
    y_pred = model.predict(X)
    return r2_score(y_pred, y)

def train(X, y, model):
    model.fit(X, y)
    return model

def get_test_data():
    data = pandas.read_csv("X_test.csv")
    X = data.loc[:, "x0":].to_numpy()
    X = impute_missing_values(X)
    X = normalize_data(X)
    X = feature_selection_regressor(X)
    return X


def get_eval_data():
    X_data = pandas.read_csv("X_train.csv")
    y_data = pandas.read_csv("y_train.csv")
    X = X_data.loc[:, "x0":].to_numpy()
    y = y_data["y"].to_numpy()
    X = impute_missing_values(X)
    X, y = remove_outliers(X, y)
    X = normalize_data(X)
    X = feature_selection_regressor(X, y)
    return X, y

def get_train_data():
    X, y = get_eval_data()
    return train_test_split(X, y, test_size=0.1)

NORMALIZER = None
def normalize_data(X):
    global NORMALIZER
    if NORMALIZER is None:
        NORMALIZER = RobustScaler()
        NORMALIZER.fit(X)
    return NORMALIZER.transform(X)

IMPUTER = None
def impute_missing_values(X):
    global IMPUTER
    if IMPUTER is None:
        # IMPUTER = IterativeImputer(initial_strategy="median")
        IMPUTER = SimpleImputer(strategy="median")
        IMPUTER.fit(X)
    return IMPUTER.transform(X)

def remove_outliers(X, y):
    model = IsolationForest(n_jobs=-1)
    model.fit(X)
    decision = model.predict(X)
    mask = 0 <= decision
    X = X[mask]
    y = y[mask]
    return X, y

SELECTOR = None
def feature_selection_percentile(X, y=None):
    global SELECTOR
    if SELECTOR is None:
        SELECTOR = SelectPercentile()
        SELECTOR.fit(X, y)
    return SELECTOR.transform(X)

def feature_selection_regressor(X, y=None):
    global SELECTOR
    if SELECTOR is None:
        selection = int(0.1 * X.shape[1])
        model = ExtraTreesRegressor()
        model.fit(X, y)
        indexed = list(zip(model.feature_importances_, range(len(model.feature_importances_))))
        SELECTOR = sorted(indexed)[-selection:]
        SELECTOR = [y for (x,y) in SELECTOR]
    X = X[:, SELECTOR]
    return X


def main():
    X, y = get_eval_data()
    model = ExtraTreesRegressor(n_estimators=300)
    scores = cross_validate(model, X, y, cv=5, scoring=make_scorer(r2_score), n_jobs=-1)
    print(scores["test_score"])
    model.fit(*get_eval_data())
    write_results(model)


if __name__ == "__main__":
    main()