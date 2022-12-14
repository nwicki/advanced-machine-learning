import sklearn.base
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, \
    HistGradientBoostingRegressor, IsolationForest, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
import pandas
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import time
import numpy
import datetime

TRAINING_DATA_X = ""


def StackedModel():
    estimators = [
        # ('kn', KNeighborsRegressor(weights='distance', p=1, n_neighbors=4, n_jobs=-1, leaf_size=1, algorithm='auto')),
        ('lgbm', LGBMRegressor(
            subsample_freq=1, subsample_for_bin=400000, subsample=0.5, reg_lambda=0.5, reg_alpha=3, num_leaves=16,
            n_jobs=-1, n_estimators=3000, min_split_gain=0.025, min_child_weight=2, min_child_samples=8, max_depth=8,
            learning_rate=0.01, colsample_bytree=0.7, boosting_type='gbdt')),
        ('gb', GradientBoostingRegressor(
            subsample=0.9, n_estimators=2250, min_samples_split=0.3, min_samples_leaf=0.05, min_impurity_decrease=0.1,
            max_depth=9, loss='squared_error', learning_rate=0.01, criterion='friedman_mse')),
        ('svr', SVR(epsilon=1.5, C=49, cache_size=2048)),
        # ('rf', RandomForestRegressor(n_estimators=500, max_depth=8, n_jobs=-1)),
        # ('xt', ExtraTreesRegressor(n_estimators=400, max_depth=8, n_jobs=-1)),
        ('xgb', XGBRegressor(subsample=0.4, reg_lambda=0.25, reg_alpha=1, n_estimators=3500,
                             max_depth=7, learning_rate=0.01, colsample_bytree=0.6, n_jobs=-1)),
        # ('ada', AdaBoostRegressor(n_estimators=2000)),
        # ('hgb', HistGradientBoostingRegressor(max_iter=550, min_samples_leaf=28, max_leaf_nodes=17)),
        # ('cat,', CatBoostRegressor(depth=6, learning_rate=0.1, l2_leaf_reg=7, logging_level='Silent')),
        # ('gp', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())),
        # ('br', BayesianRidge(alpha_1=1e-07, alpha_2=1e-05, lambda_1=1e-05, lambda_2=1e-07, n_iter=300)),
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
    X = data.iloc[:, 1:].to_numpy()
    X = feature_selection_variance(X, reset=False)
    X = impute_missing_values_simple(X, reset=False)
    X = feature_selection_regressor(X, reset=False)
    X = min_max_scale(X, reset=False)
    return X


def read_train_data():
    X_data = pandas.read_csv(TRAINING_DATA_X)
    y_data = pandas.read_csv("y_train.csv")
    X = X_data.iloc[:, 1:].to_numpy()
    y = y_data["y"].to_numpy()
    return X, y


def get_preprocessed_data():
    X, y = read_train_data()
    X = feature_selection_variance(X)
    X = impute_missing_values_simple(X)
    X = feature_selection_regressor(X, y)
    X = min_max_scale(X)
    X, y = remove_outliers(X, y)
    return X, y


def cache_impute_result():
    X_data = pandas.read_csv(TRAINING_DATA_X)
    X = X_data.iloc[:, 1:].to_numpy()
    X = impute_missing_values_iterative(X)
    pandas.DataFrame(data=X, columns=X_data.columns[1:]).to_csv("X_train_cached_estimators_100.csv", index=True, index_label='id')


def get_split_data():
    X, y = get_preprocessed_data()
    return train_test_split(X, y, test_size=0.1)


MIN_MAX_SCALER = None


def min_max_scale(X, reset=True):
    global MIN_MAX_SCALER
    if reset:
        MIN_MAX_SCALER = MinMaxScaler((-1, 1))
        MIN_MAX_SCALER.fit(X)
    return MIN_MAX_SCALER.transform(X)


ROBUST_SCALER = None


def robust_scale(X, reset=True):
    global ROBUST_SCALER
    if reset:
        ROBUST_SCALER = RobustScaler()
        ROBUST_SCALER.fit(X)
    return ROBUST_SCALER.transform(X)


SIMPLE_IMPUTER = None


def impute_missing_values_simple(X, reset=True):
    global SIMPLE_IMPUTER
    if reset:
        SIMPLE_IMPUTER = SimpleImputer(strategy="median")
        SIMPLE_IMPUTER.fit(X)
    return SIMPLE_IMPUTER.transform(X)


ITERATIVE_IMPUTER = None


def impute_missing_values_iterative(X, reset=True):
    start = time.perf_counter()
    global ITERATIVE_IMPUTER
    if reset:
        ITERATIVE_IMPUTER = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=100, n_jobs=-1), initial_strategy="median")
        ITERATIVE_IMPUTER.fit(X)
    end = time.perf_counter()
    print(f'Iterative imputation - runtime: {end-start} s')
    return ITERATIVE_IMPUTER.transform(X)


def remove_outliers(X, y, outliers=0.0625):
    model = IsolationForest(contamination=outliers, n_jobs=-1)
    model.fit(X)
    decision = model.predict(X)
    mask = 0 <= decision
    X = X[mask]
    y = y[mask]
    return X, y


SELECTOR_VARIANCE = None


def feature_selection_variance(X, threshold=0.001, reset=True):
    global SELECTOR_VARIANCE
    if reset:
        SELECTOR_VARIANCE = VarianceThreshold(threshold)
        SELECTOR_VARIANCE.fit(X)
    return SELECTOR_VARIANCE.transform(X)


SELECTOR_CORRELATION = None


def feature_selection_correlation(X, threshold=0.9, reset=True):
    global SELECTOR_CORRELATION
    if reset:
        corr_matrix = numpy.corrcoef(X, rowvar=False)
        upper = numpy.triu(corr_matrix, k=1)
        SELECTOR_CORRELATION = numpy.any(upper < threshold, axis=0)
    return X[:, SELECTOR_CORRELATION]


SELECTOR_PERCENTILE = None


def feature_selection_percentile(X, y=None, reset=True):
    global SELECTOR_PERCENTILE
    if reset:
        SELECTOR_PERCENTILE = SelectPercentile()
        SELECTOR_PERCENTILE.fit(X, y)
    return SELECTOR_PERCENTILE.transform(X)


SELECTOR_REGRESSOR = None


def feature_selection_regressor(X, y=None, threshold=0.1, reset=True):
    global SELECTOR_REGRESSOR
    if reset:
        selection = int(threshold * X.shape[1])
        model = ExtraTreesRegressor()
        model.fit(X, y)
        indexed = list(zip(model.feature_importances_, range(len(model.feature_importances_))))
        SELECTOR_REGRESSOR = sorted(indexed)[-selection:]
        SELECTOR_REGRESSOR = [y for (x, y) in SELECTOR_REGRESSOR]
    return X[:, SELECTOR_REGRESSOR]


# ('KNeighborsRegressor', 1.468204764998518, 0.46157129619317033)
# ('LGBMRegressor', 13.729332175978925, 0.5448872260418829)
# ('GradientBoostingRegressor', 17.80694640800357, 0.546633616899119)
# ('SVR', 0.19053514004917815, 0.531567645264672)
# ('RandomForestRegressor', 16.027996902004816, 0.5047773134960797)
# ('ExtraTreesRegressor', 2.9239139769924805, 0.5053379082346622)
# ('XGBRegressor', 7.853434015007224, 0.47745816217877224)
# ('AdaBoostRegressor', -1, 0.4876988170828006)
# ('MLPRegressor', -1, 0.4233371020725281)

def main():
    global TRAINING_DATA_X
    TRAINING_DATA_X = "X_train_cached_estimators_10.csv"
    print(f'Start time: {datetime.datetime.now().time()}')
    start = time.perf_counter()
    model = StackedModel()
    name = type(model).__name__
    # X, y = get_preprocessed_data()
    # scores = cross_validate(model, X, y, cv=10, scoring='r2', n_jobs=-1)["test_score"]
    # print(f'Validation Score: {sum(scores) / len(scores)}')
    model.fit(*get_preprocessed_data())
    write_results(model)
    end = time.perf_counter()
    print(f'{name} Runtime: {end - start} s')
    # model_start = time.perf_counter()
    # model = CatBoostRegressor(logging_level='Silent')
    # model.fit(X, y)
    # model_end = time.perf_counter()
    # run = model_end - model_start
    # print(f'{name} - Sample Execution Runtime: {run} s')
    # model = sklearn.base.clone(model)
    # param_dict = {
    #     'loss_function': ['RMSE'],
    #     'iterations': [10],
    #     'learning_rate': [1e-2],
    #     'l2_leaf_reg': [3],
    #     'bootstrap_type': ['Bayesian'],
    #     'max_depth': [6],
    #     'min_data_in_leaf': [1],
    #     'max_leaves': [31],
    #     'leaf_estimation_method': ['Newton'],
    # }
    # n_iter = 5
    # clf = RandomizedSearchCV(model, param_dict, n_iter=n_iter, verbose=10, n_jobs=-1, scoring='r2')
    # clf.fit(X, y)
    # end = time.perf_counter()
    # print(f'{name} - RandomizedSearch Runtime: {end - start} s')
    # print(f'{name} Best Score: {clf.best_score_}')
    # print(f'{name} Best Parameters: {clf.best_params_}')
    # stats = []
    # param_space = [0.05, 0.0625, 0.075, 0.0875, 0.1]
    # for param in param_space:
    #     X, y = get_preprocessed_data(param)
    #     model = SVR(kernel='rbf', C=9, degree=2, tol=1e-2, epsilon=1)
    #     # model = StackedModel()
    #     start = time.perf_counter()
    #     scores = cross_validate(model, X, y, cv=10, scoring='r2', n_jobs=-1)["test_score"]
    #     end = time.perf_counter()
    #     stats.append((sum(scores) / len(scores), type(model).__name__, end - start, param))
    #     print(stats[-1])
    # print(f'Max: {stats[numpy.argmax([x[0] for x in stats])]}')


if __name__ == "__main__":
    main()
