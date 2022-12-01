import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, \
    HistGradientBoostingRegressor, IsolationForest, StackingRegressor, StackingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import time
import numpy
import datetime
import biosppy.signals.ecg as ecg
from matplotlib.pyplot import plot, show
import neurokit2 as nk

TRAINING_DATA_X = ""
TEST_DATA_X = ""
TRAINING_DATA_y = ""
SCORER = make_scorer(f1_score, average='micro')


def StackedModel():
    estimators = [
        ('xgb', XGBClassifier(n_estimators=2000, max_depth=8, grow_policy='lossguide', learning_rate=0.1,
                              booster='gbtree', tree_method='gpu_hist', gamma=1e-3, min_child_weight=2,
                              subsample=0.7, sampling_method='uniform', colsample_bytree=0.6, reg_alpha=0.5,
                              reg_lambda=10, num_parallel_tree=1, n_jobs=-1)),
        ('svc', SVC(C=50.0, tol=1e-4, cache_size=1024))

    ]
    return StackingClassifier(estimators, n_jobs=-1)

def write_results(model):
    y_pred = model.predict(get_test_data())
    df = pd.DataFrame(y_pred, columns=["y"])
    df.to_csv("submission.csv", index_label="id")


def get_test_data():
    data = pd.read_csv(TEST_DATA_X)
    X = data.iloc[:, 1:].to_numpy()
    if "extracted" not in TEST_DATA_X:
        length = len(X)
        start = time.perf_counter()
        X = extract_features(X)
        print(f"Test data feature extraction: {time.perf_counter() - start} s")
        assert len(X) == length
        write_test_data(X)
    X = impute_missing_values_simple(X, reset=False)
    X = min_max_scale(X, reset=False)
    X = feature_selection_variance(X, reset=False)
    X = feature_selection_regressor(X, reset=False)
    return X


def read_train_data():
    X_data = pd.read_csv(TRAINING_DATA_X)
    y_data = pd.read_csv(TRAINING_DATA_y)
    X = X_data.iloc[:, 1:].to_numpy()
    y = y_data["y"].to_numpy()
    return X, y


def write_train_data(X, y):
    columns = [f"x{x}" for x in range(X.shape[1])]
    pd.DataFrame(data=X, columns=columns).to_csv("X_train_extracted.csv", index=True, index_label='id')
    pd.DataFrame(data=y, columns=["y"]).to_csv("y_train_extracted.csv", index=True, index_label='id')

def write_test_data(X):
    columns = [f"x{x}" for x in range(X.shape[1])]
    pd.DataFrame(data=X, columns=columns).to_csv("X_test_extracted.csv", index=True, index_label='id')


def get_preprocessed_data():
    X, y = read_train_data()
    if "extracted" not in TRAINING_DATA_X:
        start = time.perf_counter()
        X, y = extract_features(X, y)
        print(f"Train data feature extraction: {time.perf_counter() - start} s")
        write_train_data(X, y)
    X = impute_missing_values_simple(X)
    X = min_max_scale(X)
    X = feature_selection_variance(X)
    X = feature_selection_regressor(X, y)
    # X, y = remove_outliers(X, y)
    return X, y


def plot_values(values, markers=None):
    x = list(range(len(values)))
    y = values
    if markers is None:
        plot(x, y)
    else:
        plot(x, y, '-gD', markevery=markers)
    show()


def pqrst_features(ecg, ecg_points, ecg_points_adjusted, func):
    # Compress adjusted ecg_points over all signals
    ecg_points_compressed = func(ecg_points_adjusted, axis=0)
    # Use ecg_points as indices
    ecg_points = np.array(ecg_points, dtype=int)
    # R_Peak
    r_indices = ecg_points[:, 5]
    r_indices = r_indices[(0 <= r_indices) & (r_indices < len(ecg))]
    r_amplitude = func(ecg[r_indices])
    # Q_Peak
    q_indices = ecg_points[:, 4]
    q_indices = q_indices[(0 <= q_indices) & (q_indices < len(ecg))]
    if len(q_indices) == 0:
        q_amplitude = 0
    else:
        q_amplitude = func(ecg[q_indices])
    # S_Peak
    s_indices = ecg_points[:, -4]
    s_indices = s_indices[(0 <= s_indices) & (s_indices < len(ecg))]
    if len(q_indices) == 0:
        s_amplitude = 0
    else:
        s_amplitude = func(ecg[s_indices])
    # P_Onset to R_Onset
    pr_interval = ecg_points_compressed[3] - ecg_points_compressed[0]
    # P_Offset to R_Onset
    pr_segment = ecg_points_compressed[2] - ecg_points_compressed[0]
    # Q_Peak to S_Peak
    qrs_complex = ecg_points_compressed[-4] - ecg_points_compressed[4]
    # Q_Peak to T_Offset
    qt_interval = ecg_points_compressed[-1] - ecg_points_compressed[4]
    # R_Offset to T_Onset
    st_segment = ecg_points_compressed[-3] - ecg_points_compressed[-5]
    # RS Ratio
    rs_ratio = s_amplitude / r_amplitude
    # QRS amplitude
    qrs_amplitude = q_amplitude + r_amplitude + s_amplitude

    features = np.array([
        r_amplitude,
        q_amplitude,
        pr_interval,
        pr_segment,
        qrs_complex,
        qt_interval,
        st_segment,
        rs_ratio,
        qrs_amplitude
    ])
    return features


def pqrst_feature_extraction(ecg, waves, rpeaks):
    ecg_points = np.array(list(zip(
        waves["ECG_P_Onsets"],
        waves["ECG_P_Peaks"],
        waves["ECG_P_Offsets"],
        waves["ECG_R_Onsets"],
        waves["ECG_Q_Peaks"],
        rpeaks["ECG_R_Peaks"],
        waves["ECG_R_Offsets"],
        waves["ECG_S_Peaks"],
        waves["ECG_T_Onsets"],
        waves["ECG_T_Peaks"],
        waves["ECG_T_Offsets"])))

    ecg_points[:, np.all(np.isnan(ecg_points), axis=0)] = 0
    feature_median = np.nanmedian(ecg_points, axis=0)
    indices = np.where(np.isnan(ecg_points))
    ecg_points[indices] = np.take(feature_median, indices[1])

    rpeaks = ecg_points[:, 5]
    rr_interval = rpeaks[1:] - rpeaks[:-1]
    rr_interval = np.array([
        np.mean(rr_interval),
        np.median(rr_interval),
        np.min(rr_interval),
        np.max(rr_interval),
        np.std(rr_interval)
    ])

    ref_r = ecg_points[0, 5]
    r_offset = np.repeat(rpeaks.reshape(-1, 1) - ref_r, ecg_points.shape[1], axis=1)
    ecg_points_adjusted = ecg_points - r_offset
    ecg_points_adjusted = impute_mean_values(ecg_points_adjusted)
    ecg_points_adjusted = ecg_points_adjusted[local_outlier_detection(ecg_points_adjusted)]

    features = np.concatenate((
        pqrst_features(ecg, ecg_points, ecg_points_adjusted, np.mean),
        pqrst_features(ecg, ecg_points, ecg_points_adjusted, np.median),
        pqrst_features(ecg, ecg_points, ecg_points_adjusted, np.min),
        pqrst_features(ecg, ecg_points, ecg_points_adjusted, np.max),
        pqrst_features(ecg, ecg_points, ecg_points_adjusted, np.std),
        rr_interval
    ))

    return features


def HRV_feature_extraction(hrv_indices):
    features = [
        hrv_indices["HRV_MeanNN"],
        hrv_indices["HRV_SDNN"],
        hrv_indices["HRV_RMSSD"],
        hrv_indices["HRV_MedianNN"],
        hrv_indices["HRV_pNN50"],
        hrv_indices["HRV_MinNN"],
        hrv_indices["HRV_MaxNN"],
        hrv_indices["HRV_LF"],
        hrv_indices["HRV_HF"],
        hrv_indices["HRV_LFHF"],
        hrv_indices["HRV_LnHF"],
    ]
    features = [x.iloc[0] for x in features]
    return np.array(features)


def extract_features(X, Y = None):
    X_new = []
    Y_new = []
    sampling_rate = 300
    for i, x in enumerate(X):
        print(f"Process sample: {i}")
        x = x[~np.isnan(x)]

        # Neurokit Feature Extraction
        ecg_cleaned = np.array(nk.ecg_clean(x, sampling_rate))
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate)
        if len(rpeaks["ECG_R_Peaks"]) == 0:
            X_new.append(None)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate)
            except (KeyError, ValueError):
                X_new.append(None)
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Heart Rate Variability in time, frequency, and non-linear domain
                hrv_indices = nk.hrv(rpeaks, sampling_rate)
            except (KeyError, ValueError):
                X_new.append(None)
                continue

        features = pqrst_feature_extraction(ecg_cleaned, waves, rpeaks)
        features = np.concatenate((features, HRV_feature_extraction(hrv_indices)))

        X_new.append(features)
        if Y is not None:
            Y_new.append(Y[i])

    # For all cases where feature extraction failed (features == None),
    # use the features of the closest signal as reference
    for i, x_new in enumerate(X_new):
        if x_new is None:
            closest = 0
            distance = 2e28
            X_i = X[i]
            for j, x in enumerate(X):
                if X_new[j] is not None:
                    dist = np.sum(np.abs(x - X_i))
                    if dist < distance:
                        distance = dist
                        closest = j
            X_new[i] = X_new[closest]

    X_new = np.array(X_new)

    if Y is None:
        return X_new

    return X_new, np.array(Y_new)


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


def impute_mean_values(X):
    model = SimpleImputer(strategy="mean")
    return model.fit_transform(X)

SIMPLE_IMPUTER = None


def impute_missing_values_simple(X, reset=True):
    global SIMPLE_IMPUTER
    if reset:
        SIMPLE_IMPUTER = SimpleImputer(strategy="median")
        SIMPLE_IMPUTER.fit(X)
    return SIMPLE_IMPUTER.transform(X)


def local_outlier_detection(X):
    model = LocalOutlierFactor(n_neighbors=min(len(X), 20), n_jobs=-1)
    decision = model.fit_predict(X)
    mask = 0 <= decision
    return mask


def remove_outliers(X, y, outliers=0.5):
    model = IsolationForest(contamination=outliers, n_jobs=-1)
    model.fit(X)
    decision = model.predict(X)
    mask = 0 <= decision
    X = X[mask]
    y = y[mask]
    return X, y


SELECTOR_VARIANCE = None


def feature_selection_variance(X, threshold=0.01, reset=True):
    global SELECTOR_VARIANCE
    if reset:
        SELECTOR_VARIANCE = VarianceThreshold(threshold)
        SELECTOR_VARIANCE.fit(X)
    return SELECTOR_VARIANCE.transform(X)


SELECTOR_REGRESSOR = None


def feature_selection_regressor(X, y=None, threshold=0.475, reset=True):
    global SELECTOR_REGRESSOR
    if reset:
        selection = int(threshold * X.shape[1])
        model = ExtraTreesRegressor()
        model.fit(X, y)
        indexed = list(zip(model.feature_importances_, range(len(model.feature_importances_))))
        SELECTOR_REGRESSOR = sorted(indexed)[-selection:]
        SELECTOR_REGRESSOR = [y for (x, y) in SELECTOR_REGRESSOR]
    return X[:, SELECTOR_REGRESSOR]


# Thoughts:
# Probably cannot use outlier detection as it removes samples from classes with fewer occurrences
# Feature selection should be fine though^^
# Model can still be improved drastically
# model = SVC(C=50.0, tol=1e-4, cache_size=1024) -> Validation Score: 0.7793457623890372
# model = XGBClassifier(n_estimators=2000, max_depth=8, grow_policy= 'lossguide', learning_rate=0.1,
#                       booster='gbtree', tree_method='gpu_hist', gamma=1e-3, min_child_weight=2,
#                       subsample=0.7, sampling_method='uniform', colsample_bytree=0.6, reg_alpha=0.5,
#                       reg_lambda=10, num_parallel_tree=1, n_jobs=-1) -> Validation Score: 0.7871956816427264

def main():
    global TRAINING_DATA_X
    # TRAINING_DATA_X = "X_train.csv"
    TRAINING_DATA_X = "X_train_extracted.csv"
    # TRAINING_DATA_X = "X_train_partial.csv"
    # TRAINING_DATA_X = "X_train_extracted_partial.csv"
    global TRAINING_DATA_y
    # TRAINING_DATA_y = "y_train.csv"
    TRAINING_DATA_y = "y_train_extracted.csv"
    # TRAINING_DATA_y = "y_train_partial.csv"
    # TRAINING_DATA_y = "y_train_extracted_partial.csv"
    global TEST_DATA_X
    TEST_DATA_X = "X_test_extracted.csv"
    # TEST_DATA_X = "X_test_extracted.csv"

    print(f'Start time: {datetime.datetime.now()}')
    start = time.perf_counter()
    model = StackedModel()
    X, y = get_preprocessed_data()
    # scores = cross_validate(model, X, y, cv=5, scoring=SCORER, n_jobs=-1)["test_score"]
    # print(f'Validation Score: {sum(scores) / len(scores)}')
    model.fit(X, y)
    write_results(model)
    end = time.perf_counter()
    print(f'End time: {datetime.datetime.now()}')
    print(f'Runtime: {end - start} s')

    # start = time.perf_counter()
    # X, y = get_preprocessed_data()
    # model = XGBClassifier()
    # name = type(model).__name__
    # model.fit(X, y)
    # end = time.perf_counter()
    # print(f'{name} - Sample Execution Runtime: {end - start} s')
    # model = clone(model)
    # param_dict = {
    #     'n_estimators': [2000],
    #     'max_depth': [4, 5, 6, 7, 8],
    #     'grow_policy': ['lossguide'],
    #     'learning_rate': [0.1],
    #     'booster': ['gbtree'],
    #     'tree_method': ['gpu_hist'],
    #     'gamma': [1e-3],
    #     'min_child_weight': [2],
    #     'subsample': [0.7],
    #     'sampling_method': ['uniform'],
    #     'colsample_bytree': [0.6],
    #     'reg_alpha': [0.5],
    #     'reg_lambda': [10],
    #     'num_parallel_tree': [1],
    #     'n_jobs': [-1],
    #     'random_state': [42],
    # }
    # print(f"Parameter combinations: {np.prod([len(v) for k, v in param_dict.items()])}")
    # n_iter = 20
    # clf = RandomizedSearchCV(model, param_dict, n_iter=n_iter, verbose=10, n_jobs=-1, scoring=SCORER)
    # clf.fit(X, y)
    # end = time.perf_counter()
    # print(f'{name} - RandomizedSearch Runtime: {end - start} s')
    # print(f'{name} Best Score: {clf.best_score_}')
    # print(f'{name} Best Parameters: {clf.best_params_}')

    # stats = []
    # param_space = [0.4, 0.45, 0.5, 0.6]
    # for param in param_space:
    #     X, y = get_preprocessed_data(param, False)
    #     model = SVC()
    #     start = time.perf_counter()
    #     scores = cross_validate(model, X, y, cv=5, scoring=SCORER, n_jobs=-1)["test_score"]
    #     end = time.perf_counter()
    #     stats.append((sum(scores) / len(scores), end - start, param))
    #     print(stats[-1])
    # print(f'Max: {stats[numpy.argmax([x[0] for x in stats])]}')


if __name__ == "__main__":
    main()
6