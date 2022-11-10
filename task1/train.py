import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RANSACRegressor
from sklearn.svm import OneClassSVM, SVR, LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, IsolationForest, ExtraTreesClassifier, StackingRegressor
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE, SelectPercentile
from scipy.stats import rankdata
import pandas
import numpy
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import time

TRAINING_DATA_X = ""
def StackedModel():
        estimators = [
            ('kn', KNeighborsRegressor(n_neighbors=12, weights='distance', leaf_size=1, p=1)),
            ('lgbm', LGBMRegressor(boosting_type='gbdt', num_leaves=32, max_depth=8, n_estimators=2500)),
            ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=4)),
            ('svr', SVR(kernel='rbf', C=9, degree=2, tol=1e-2, epsilon=1)),
            ('rf', RandomForestRegressor(n_estimators=500, max_depth=8, n_jobs=-1)),
            ('xt', ExtraTreesRegressor(n_estimators=400, max_depth=8, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=400, max_depth=4, n_jobs=-1)),
            ('ada', AdaBoostRegressor(n_estimators=2000)),
            ('mlp', MLPRegressor(alpha=1e-1, early_stopping=True, hidden_layer_sizes=(7, 7, 7, 7, 7, 7, 7, 7), learning_rate='invscaling', learning_rate_init=0.1, max_iter=int(1e4)))
             ]
        return StackingRegressor(estimators=estimators, cv=10, n_jobs=-1)

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
    X = impute_missing_values_simple(X)
    X = normalize_data(X)
    X = feature_selection_regressor(X, reset=False)
    return X


def read_data():
    X_data = pandas.read_csv(TRAINING_DATA_X)
    y_data = pandas.read_csv("y_train.csv")
    X = X_data.loc[:, "x0":].to_numpy()
    y = y_data["y"].to_numpy()
    return X, y

def get_eval_data():
    X, y = read_data()
    X = impute_missing_values_simple(X)
    X, y = remove_outliers(X, y)
    X = normalize_data(X)
    X = feature_selection_regressor(X, y)
    return X, y

def cache_impute_result():
    X_data = pandas.read_csv(TRAINING_DATA_X)
    X = X_data.loc[:, "x0":].to_numpy()
    X = impute_missing_values_iterative(X)
    pandas.DataFrame(data=X, columns=X_data.columns[1:]).to_csv("X_train_cached.csv", index=True, index_label='id')

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
def impute_missing_values_simple(X):
    global IMPUTER
    if IMPUTER is None:
        # IMPUTER = IterativeImputer(initial_strategy="median")
        IMPUTER = SimpleImputer(strategy="median")
        IMPUTER.fit(X)
    return IMPUTER.transform(X)

def impute_missing_values_iterative(X):
    global IMPUTER
    if IMPUTER is None:
        IMPUTER = IterativeImputer(initial_strategy="median")
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
def feature_selection_percentile(X, y=None, reset=True):
    global SELECTOR
    if reset:
        SELECTOR = SelectPercentile()
        SELECTOR.fit(X, y)
    return SELECTOR.transform(X)

def feature_selection_regressor(X, y=None, features=0.1125, reset=True):
    global SELECTOR
    if reset:
        selection = int(features * X.shape[1])
        model = ExtraTreesRegressor()
        model.fit(X, y)
        indexed = list(zip(model.feature_importances_, range(len(model.feature_importances_))))
        SELECTOR = sorted(indexed)[-selection:]
        SELECTOR = [y for (x,y) in SELECTOR]
    return X[:, SELECTOR]


# ('KNeighborsRegressor', 1.468204764998518, 0.46157129619317033)
# ('LGBMRegressor', 13.729332175978925, 0.5448872260418829)
# ('GradientBoostingRegressor', 17.80694640800357, 0.546633616899119)
# ('SVR', 0.19053514004917815, 0.531567645264672)
# ('RandomForestRegressor', 16.027996902004816, 0.5047773134960797)
# ('ExtraTreesRegressor', 2.9239139769924805, 0.5053379082346622)
# ('XGBRegressor', 7.853434015007224, 0.47745816217877224)
# ('AdaBoostRegressor', -1, 0.4876988170828006)
# ('MLPRegressor', -1, 0.4233371020725281)

# StackingRegressor SimpleImputer
# 727.248965251958
# 0.5576333296254736

# StackingRegressor IterativeImputer
# 778.5039008309832
# 0.48661298297559813


def main():
    global TRAINING_DATA_X
    TRAINING_DATA_X = "X_train.csv"
    X, y = read_data()
    model = StackedModel()
    start = time.perf_counter()
    scores = cross_validate(model, X, y, cv=10, scoring='r2', n_jobs=-1)["test_score"]
    end = time.perf_counter()
    print(end - start)
    print(sum(scores) / len(scores))
    # , 'poly' 'degree': [2, 3, 4, 5] 'coef0': [0], 'rbf', 'sigmoid' 'coef0': [0]
    # model = SVR()
    # clf = GridSearchCV(model, { 'kernel': ['rbf'], 'C': [19], 'gamma': ['scale'], 'tol': [1e-2, 1e-3, 1e-4], 'epsilon': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]}, verbose=10, n_jobs=-1, scoring='r2')
    # clf.fit(X, y)
    # print(type(model).__name__, clf.best_score_)
    # print(type(model).__name__, clf.best_params_)
    # stats = []
    # param_space = [0.10625, 0.1125, 0.11875, 0.125]
    # for param in param_space:
    #     X, y = get_eval_data(param)
    #     model = SVR(kernel='rbf', C=9, degree=2, tol=1e-2, epsilon=1)
    #     # model = StackedModel()
    #     start = time.perf_counter()
    #     scores = cross_validate(Pmodel, X, y, cv=10, scoring='r2', n_jobs=-1)["test_score"]
    #     end = time.perf_counter()
    #     stats.append((sum(scores) / len(scores), type(model).__name__, end - start, param))
    #     print(stats[-1])
    # print(f'Max: {stats[numpy.argmax([x[0] for x in stats])]}')
    model = StackedModel()
    model.fit(*get_eval_data())
    write_results(model)



if __name__ == "__main__":
    main()