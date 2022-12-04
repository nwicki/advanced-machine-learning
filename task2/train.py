import pickle
import warnings
import json
import heartpy.exceptions
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, VotingClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import clone
import time
import numpy
import datetime
import biosppy.signals.ecg as ecg
from matplotlib.pyplot import plot, show
import neurokit2 as nk
import heartpy as hp

TRAINING_DATA_X = ""
TEST_DATA_X = ""
TRAINING_DATA_y = ""
SCORER = make_scorer(f1_score, average='micro')


def XGB_Classifier():
    return XGBClassifier(n_estimators=1000, max_depth=5, grow_policy='lossguide', learning_rate=0.1,
                         tree_method='gpu_hist', gamma=1e-3, min_child_weight=2, subsample=0.7,
                         sampling_method='uniform', colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=10,
                         num_parallel_tree=1, n_jobs=-1, random_state=42)


def LGBM_Classifier():
    return LGBMClassifier(max_depth=6, learning_rate=0.01, n_estimators=1500, min_split_gain=0.1,
                          min_child_weight=0.01, min_child_samples=40, subsample=0.6, subsample_freq=5,
                          colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=0.5, n_jobs=-1, random_state=42)


def CatBoost_Classifier():
    return CatBoostClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1,
                              task_type="GPU", gpu_ram_part=0.2, verbose=False)


def VotingModel():
    estimators = [
        ('xgb', XGB_Classifier()),
        # ('svc', SVC(C=50.0, tol=1e-4, cache_size=1024, class_weight='balanced'))
        ('lgbm', LGBM_Classifier()),
        # ('cat', CatBoostClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1,
        #                            task_type="GPU", gpu_ram_part=0.2, verbose=False))

    ]
    return VotingClassifier(estimators)


def write_results(model):
    y_pred = model.predict(get_test_data())
    df = pd.DataFrame(y_pred, columns=["y"])
    df.to_csv("submission.csv", index_label="id")


def write_train_data(X, y):
    columns = [f"x{x}" for x in range(X.shape[1])]
    pd.DataFrame(data=X, columns=columns).to_csv("X_train_extracted.csv", index=True, index_label='id')


def write_test_data(X):
    columns = [f"x{x}" for x in range(X.shape[1])]
    pd.DataFrame(data=X, columns=columns).to_csv("X_test_extracted.csv", index=True, index_label='id')


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
    X = standard_scale(X, reset=False)
    X = feature_selection_variance(X, reset=False)
    X = feature_selection_regressor(X, reset=False)
    return X


def read_train_data():
    X_data = pd.read_csv(TRAINING_DATA_X)
    y_data = pd.read_csv(TRAINING_DATA_y)
    X = X_data.iloc[:, 1:].to_numpy()
    y = y_data["y"].to_numpy()
    return X, y


def get_train_data(param=None):
    X, y = read_train_data()
    if "extracted" not in TRAINING_DATA_X:
        start = time.perf_counter()
        X = extract_features(X)
        print(f"Train data feature extraction: {time.perf_counter() - start} s")
        write_train_data(X, y)
    X = impute_missing_values_simple(X)
    X = standard_scale(X)
    X = feature_selection_variance(X)
    X = feature_selection_regressor(X, y)
    return X, y


def plot_values(values, markers=None):
    x = list(range(len(values)))
    y = values
    if markers is None:
        plot(x, y)
    else:
        plot(x, y, '-gD', markevery=markers)
    show()


def get_amplitude(ecg_signal, ecg_points, func, index):
    indices = ecg_points[:, index]
    indices = indices[(0 <= indices) & (indices < len(ecg_signal))]
    if len(indices) == 0:
        return 0
    return func(ecg_signal[indices])


def get_statistical_values(array):
    return np.array([
        np.mean(array),
        np.median(array),
        np.min(array),
        np.max(array),
        np.std(array)
    ])


def get_interval(peaks):
    if 1 < len(peaks):
        interval = peaks[1:] - peaks[:-1]
        return get_statistical_values(interval)
    return np.zeros(5)


def biosppy_feature_extraction(signal, sampling_rate):
    r_peaks, = ecg.engzee_segmenter(signal, sampling_rate)
    beats, _ = ecg.extract_heartbeats(signal, r_peaks, sampling_rate)

    heartbeat_len = 180
    substitute = np.array(heartbeat_len * [np.nan])

    try:
        heartbeat_mean = np.mean(beats, axis=0)
    except ValueError:
        heartbeat_mean = substitute
    try:
        heartbeat_median = np.median(beats, axis=0)
    except ValueError:
        heartbeat_median = substitute
    try:
        heartbeat_min = np.min(beats, axis=0)
    except ValueError:
        heartbeat_min = substitute
    try:
        heartbeat_max = np.max(beats, axis=0)
    except ValueError:
        heartbeat_max = substitute
    try:
        heartbeat_std = np.std(beats, axis=0)
    except ValueError:
        heartbeat_std = substitute

    if type(heartbeat_mean) is not np.ndarray or len(heartbeat_mean) != heartbeat_len:
        heartbeat_mean = substitute
    if type(heartbeat_median) is not np.ndarray or len(heartbeat_median) != heartbeat_len:
        heartbeat_median = substitute
    if type(heartbeat_min) is not np.ndarray or len(heartbeat_min) != heartbeat_len:
        heartbeat_min = substitute
    if type(heartbeat_max) is not np.ndarray or len(heartbeat_max) != heartbeat_len:
        heartbeat_max = substitute
    if type(heartbeat_std) is not np.ndarray or len(heartbeat_std) != heartbeat_len:
        heartbeat_std = substitute

    return np.concatenate((heartbeat_mean, heartbeat_median, heartbeat_min, heartbeat_max, heartbeat_std)), r_peaks


def PQRST_features(ecg_signal, ecg_points, ecg_points_adjusted, func):
    # Compress adjusted ecg_points over all signals
    ecg_points_compressed = func(ecg_points_adjusted, axis=0)
    # Use ecg_points as indices
    ecg_points = np.array(ecg_points, dtype=int)

    # Get amplitudes cleaned
    amplitudes = np.array([get_amplitude(ecg_signal, ecg_points, func, x) for x in range(ecg_points.shape[1])])

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
    rs_ratio = amplitudes[7] / amplitudes[5]
    # RQ Ratio
    rq_ratio = amplitudes[4] / amplitudes[5]
    # QRS amplitude
    qrs_amplitude = amplitudes[4] + amplitudes[5] + amplitudes[6]

    features = np.array([
        pr_interval,
        pr_segment,
        qrs_complex,
        qt_interval,
        st_segment,
        rs_ratio,
        rq_ratio,
        qrs_amplitude
    ])
    return np.concatenate((amplitudes, features))


def PQRST_feature_extraction(ecg, waves, rpeaks):
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

    if 0 < np.sum(np.isnan(ecg_points)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_median = np.nanmedian(ecg_points, axis=0)
        indices = np.where(np.isnan(ecg_points))
        ecg_points[indices] = np.take(feature_median, indices[1])

    intervals = np.concatenate([get_interval(ecg_points[:, x]) for x in range(ecg_points.shape[1])])

    ref_r = ecg_points[0, 5]
    r_offset = np.repeat(ecg_points[:, 5].reshape(-1, 1) - ref_r, ecg_points.shape[1], axis=1)
    ecg_points_adjusted = ecg_points - r_offset

    features = np.concatenate((
        PQRST_features(ecg, ecg_points, ecg_points_adjusted, np.mean),
        PQRST_features(ecg, ecg_points, ecg_points_adjusted, np.median),
        PQRST_features(ecg, ecg_points, ecg_points_adjusted, np.min),
        PQRST_features(ecg, ecg_points, ecg_points_adjusted, np.max),
        PQRST_features(ecg, ecg_points, ecg_points_adjusted, np.std),
        intervals
    ))

    return features


def HRV_feature_extraction(hrv_indices):
    features = [
        hrv_indices["HRV_MeanNN"],
        hrv_indices["HRV_SDNN"],
        hrv_indices["HRV_SDANN1"],
        hrv_indices["HRV_SDNNI1"],
        hrv_indices["HRV_SDANN2"],
        hrv_indices["HRV_SDNNI2"],
        hrv_indices["HRV_SDANN5"],
        hrv_indices["HRV_SDNNI5"],
        hrv_indices["HRV_RMSSD"],
        hrv_indices["HRV_SDSD"],
        hrv_indices["HRV_CVNN"],
        hrv_indices["HRV_CVSD"],
        hrv_indices["HRV_MedianNN"],
        hrv_indices["HRV_MadNN"],
        hrv_indices["HRV_MCVNN"],
        hrv_indices["HRV_IQRNN"],
        hrv_indices["HRV_Prc20NN"],
        hrv_indices["HRV_Prc80NN"],
        hrv_indices["HRV_pNN50"],
        hrv_indices["HRV_pNN20"],
        hrv_indices["HRV_MinNN"],
        hrv_indices["HRV_MaxNN"],
        hrv_indices["HRV_HTI"],
        hrv_indices["HRV_TINN"],
        hrv_indices["HRV_ULF"],
        hrv_indices["HRV_VLF"],
        hrv_indices["HRV_LF"],
        hrv_indices["HRV_HF"],
        hrv_indices["HRV_VHF"],
        hrv_indices["HRV_LFHF"],
        hrv_indices["HRV_LFn"],
        hrv_indices["HRV_HFn"],
        hrv_indices["HRV_LnHF"],
        hrv_indices["HRV_SD1"],
        hrv_indices["HRV_SD2"],
        hrv_indices["HRV_SD1SD2"],
        hrv_indices["HRV_S"],
        hrv_indices["HRV_CSI"],
        hrv_indices["HRV_CVI"],
        hrv_indices["HRV_CSI_Modified"],
        hrv_indices["HRV_PIP"],
        hrv_indices["HRV_IALS"],
        hrv_indices["HRV_PSS"],
        hrv_indices["HRV_PAS"],
        hrv_indices["HRV_GI"],
        hrv_indices["HRV_SI"],
        hrv_indices["HRV_AI"],
        hrv_indices["HRV_PI"],
        hrv_indices["HRV_C1d"],
        hrv_indices["HRV_C1a"],
        hrv_indices["HRV_SD1d"],
        hrv_indices["HRV_SD1a"],
        hrv_indices["HRV_C2d"],
        hrv_indices["HRV_C2a"],
        hrv_indices["HRV_SD2d"],
        hrv_indices["HRV_SD2a"],
        hrv_indices["HRV_Cd"],
        hrv_indices["HRV_Ca"],
        hrv_indices["HRV_SDNNd"],
        hrv_indices["HRV_SDNNa"],
        hrv_indices["HRV_DFA_alpha1"],
        hrv_indices["HRV_MFDFA_alpha1_Width"],
        hrv_indices["HRV_MFDFA_alpha1_Peak"],
        hrv_indices["HRV_MFDFA_alpha1_Mean"],
        hrv_indices["HRV_MFDFA_alpha1_Max"],
        hrv_indices["HRV_MFDFA_alpha1_Delta"],
        hrv_indices["HRV_MFDFA_alpha1_Asymmetry"],
        hrv_indices["HRV_MFDFA_alpha1_Fluctuation"],
        hrv_indices["HRV_MFDFA_alpha1_Increment"],
        hrv_indices["HRV_ApEn"],
        hrv_indices["HRV_SampEn"],
        hrv_indices["HRV_ShanEn"],
        hrv_indices["HRV_FuzzyEn"],
        hrv_indices["HRV_MSEn"],
        hrv_indices["HRV_CMSEn"],
        hrv_indices["HRV_RCMSEn"],
        hrv_indices["HRV_CD"],
        hrv_indices["HRV_HFD"],
        hrv_indices["HRV_KFD"],
        hrv_indices["HRV_LZC"],
    ]
    features = [x.iloc[0] for x in features]
    return np.array(features)


def neurokit_feature_extraction(signal, sampling_rate, biosppy_r_peaks):
    ecg_cleaned = nk.ecg_clean(signal, sampling_rate)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate)

    if len(rpeaks["ECG_R_Peaks"]) == 0:
        if len(biosppy_r_peaks) != 0:
            rpeaks["ECG_R_Peaks"] = biosppy_r_peaks
        else:
            rpeaks["ECG_R_Peaks"] = np.array(180 * [np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate)
        except Exception:
            waves = json.load(open("waves.txt"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Heart Rate Variability in time, frequency, and non-linear domain
            hrv_indices = nk.hrv(rpeaks, sampling_rate)
        except Exception:
            hrv_indices = pd.read_csv("hrv_indices.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            heartbeat_length = np.array([len(nk.ecg_segment(ecg_cleaned, sampling_rate=sampling_rate)["1"])])
        except Exception:
            heartbeat_length = np.array([np.nan])
        try:
            quality = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate)
        except Exception:
            quality = np.array([np.nan])
        try:
            rate = nk.ecg_rate(ecg_cleaned, sampling_rate)
        except Exception:
            rate = np.array([np.nan])
        try:
            rsp = nk.ecg_rsp(rate, sampling_rate)
        except Exception:
            rsp = np.array([np.nan])

    return np.concatenate((
        heartbeat_length,
        get_statistical_values(quality),
        get_statistical_values(rate),
        get_statistical_values(rsp),
        PQRST_feature_extraction(ecg_cleaned, waves, rpeaks),
        HRV_feature_extraction(hrv_indices)))

def heartpy_feature_extraction(signal, sampling_rate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, measures = hp.process(signal, sampling_rate)
            return np.array([v for k, v in measures.items()])
        except heartpy.exceptions.BadSignalWarning:
            return pickle.load(open("heartpy_features.pickle", "rb"))


def extract_features(X):
    X_new = []
    sampling_rate = 300
    for i, x in enumerate(X):
        print(f"Process sample: {i}")
        x = x[~np.isnan(x)]

        # Heartpy Feature Extraction
        heartpy_features = heartpy_feature_extraction(x, sampling_rate)

        # Biosppy Feature Extraction
        biosppy_features, r_peaks = biosppy_feature_extraction(x, sampling_rate)

        # Neurokit Feature Extraction
        neurokit_features = neurokit_feature_extraction(x, sampling_rate, r_peaks)

        features = np.concatenate((
            biosppy_features,
            neurokit_features,
            heartpy_features
        ))

        X_new.append(features)

    X_new = np.array(X_new)

    X_new[np.isinf(X_new)] = np.nan

    return X_new


MIN_MAX_SCALER = None


def min_max_scale(X, interval=(-10, 10), reset=True):
    global MIN_MAX_SCALER
    if reset:
        MIN_MAX_SCALER = MinMaxScaler(interval)
        MIN_MAX_SCALER.fit(X)
    return MIN_MAX_SCALER.transform(X)


STANDARD_SCALER = None


def standard_scale(X, reset=True):
    global STANDARD_SCALER
    if reset:
        STANDARD_SCALER = StandardScaler(with_mean=False, with_std=False)
        STANDARD_SCALER.fit(X)
    return STANDARD_SCALER.transform(X)


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


def impute_missing_values_simple(X, strategy="median", reset=True):
    global SIMPLE_IMPUTER
    if reset:
        SIMPLE_IMPUTER = SimpleImputer(strategy=strategy)
        SIMPLE_IMPUTER.fit(X)
    return SIMPLE_IMPUTER.transform(X)


def local_outlier_detection(X):
    model = LocalOutlierFactor(n_neighbors=min(len(X), 20), n_jobs=-1)
    decision = model.fit_predict(X)
    mask = 0 <= decision
    return mask


SELECTOR_VARIANCE = None


def feature_selection_variance(X, threshold=0.01, reset=True):
    global SELECTOR_VARIANCE
    if reset:
        SELECTOR_VARIANCE = VarianceThreshold(threshold)
        SELECTOR_VARIANCE.fit(X)
    return SELECTOR_VARIANCE.transform(X)


SELECTOR_REGRESSOR = None


def feature_selection_regressor(X, y=None, threshold=0.12, reset=True):
    global SELECTOR_REGRESSOR
    if reset:
        selection = int(threshold * X.shape[1])
        model = LGBMRegressor()
        model.fit(X, y)
        indexed = list(zip(model.feature_importances_, range(len(model.feature_importances_))))
        SELECTOR_REGRESSOR = sorted(indexed)[-selection:]
        SELECTOR_REGRESSOR = [y for (x, y) in SELECTOR_REGRESSOR]
    return X[:, SELECTOR_REGRESSOR]


def model_search(model, param_dict, n_iter=20):
    start = time.perf_counter()
    X, y = get_train_data()
    name = type(model).__name__
    model.fit(X, y)
    end = time.perf_counter()
    print(f'{name} - Sample Execution Runtime: {end - start} s')
    model = clone(model)
    print(f"Parameter combinations: {np.prod([len(v) for k, v in param_dict.items()])}")
    clf = RandomizedSearchCV(model, param_dict, n_iter=n_iter, verbose=10, n_jobs=1, scoring=SCORER)
    clf.fit(X, y)
    end = time.perf_counter()
    print(f'{name} - RandomizedSearch Runtime: {end - start} s')
    print(f'{name} Best Score: {clf.best_score_}')
    print(f'{name} Best Parameters: {clf.best_params_}')


def test_model(model=None, model_path=None):
    start = time.perf_counter()
    if model is None:
        if model_path is None:
            print(f"No model path provided")
        model = pickle.load(open(model_path, "rb"))
    else:
        X, y = get_train_data()
        model.fit(X, y)
        pickle.dump(model, open("model.pickle", "wb"))
    write_results(model)
    end = time.perf_counter()
    print(f'Testing Runtime: {end - start} s')


def validate_model(model):
    print(f"Validating model: {type(model).__name__}")
    start = time.perf_counter()
    X, y = get_train_data()
    scores = cross_validate(model, X, y, cv=5, scoring=SCORER, n_jobs=-1)["test_score"]
    print(f'Validation Score: {sum(scores) / len(scores)}')
    end = time.perf_counter()
    print(f'Validation Runtime: {end - start} s')


def validate_test_model(model):
    start = time.perf_counter()
    validate_model(model)
    test_model(model)
    end = time.perf_counter()
    print(f'Validation & Testing Runtime: {end - start} s')


def param_search(model, param_space):
    start = time.perf_counter()
    stats = []
    for param in param_space:
        model_start = time.perf_counter()
        X, y = get_train_data(param)
        model = clone(model)
        scores = cross_validate(model, X, y, cv=5, scoring=SCORER, n_jobs=-1)["test_score"]
        model_end = time.perf_counter()
        stats.append((sum(scores) / len(scores), model_end - model_start, param))
        print(stats[-1])
    print(f'Max: {stats[numpy.argmax([x[0] for x in stats])]}')
    end = time.perf_counter()
    print(f'Runtime: {end - start} s')


def main():
    global TRAINING_DATA_X
    # TRAINING_DATA_X = "X_train.csv"
    TRAINING_DATA_X = "X_train_extracted.csv"
    global TRAINING_DATA_y
    TRAINING_DATA_y = "y_train.csv"
    global TEST_DATA_X
    TEST_DATA_X = "X_test.csv"
    # TEST_DATA_X = "X_test_extracted.csv"

    print(f'Start time: {datetime.datetime.now()}')

    model = LGBM_Classifier()
    test_model(model)

    # param_search(model, [0.11, 0.12, 0.13, 0.14])

    # model = CatBoostClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, task_type="GPU", gpu_ram_part=0.2, verbose=False)
    #
    # param_dict = {
    #     "n_estimators": [100, 500, 1000],
    #     "max_depth": [4, 6, 8],
    #     "learning_rate": [0.1, 0.01, 0.001],
    #     "task_type": ["GPU"],
    #     "gpu_ram_part": [0.2],
    #     "random_state": [42]
    # }
    #
    # model_search(model, param_dict, 20)
    #
    print(f'End time: {datetime.datetime.now()}')

if __name__ == "__main__":
    main()
