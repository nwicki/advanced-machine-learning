# Advanced Machine Learning

## Task 1: PREDICT THE AGE OF A BRAIN FROM MRI FEATURES
**1. Imputation:** As the dataset provided contains missing values, Iterative Imputation was used. The estimator of choice was the ExtraTrees Regressor with 10
estimators. As the computation is done over and over again, the runtime is considerable and caching the result is advised.

**2. Feature selection:** To reduce the complexity of the problem, features were carefully selected. To do this, features with low variance were removed using
the Variance Threshold interface (Threshold of le-3). To additionally pick the best features, an ExtraTreesRegressor base model was trained on the
dataset and target values to predict feature importance values, of which the 10% most important features were chosen.
Standardization: To make sure the features are scaled to the same range of values and contribute equally, the MinMaxScaler was used with the interval
[−1, 1].

**3. Outlier removal:** As the dataset contains outliers and as they can negatively impact the interpretability of the data, the IsolationForest method was used
with an assumed contamination of 6.25%.

**4. Regression:** For the final regression task, a Stacked Regressor was used made up from estimators such as the XGBRegressor, LGBMRegressor,
GradientBoostingRegressor, and SVR.

**Parameter search:** The used values were computed using GridSearchCV and Randomized SearchCV by heavily testing various combinations of
parameters to make sure to pick the best combination.
Overfitting: Prevented by cross validation and regularization

## Task 2: HEART RHYTHM CLASSIFICATION FROM RAW ECG SIGNALS
**1. Feature extraction:**
- Use [BioSPPy](https://biosppy.readthedocs.io/en/stable/) to extract all heartbeats from signals (convenient as all extracted heartbeats are mapped to a fixed length interval)
- Use [NeuroKit2](https://neuropsychology.github.io/NeuroKit/) to extract heartbeat length, quality of signal, heart rate, respiratory features, PQRST features (such as the intervals between peaks,
amplitudes of peaks, special segments and intervals e.g. PR interval - PR segment - QT Interval - etc., ratios between peaks, etc.), and heart rate
variability features in the time, frequency, and non-linear domain based on FFT
- Use [HeartPy](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/) to extract additional features such as breathing rate and some additional heart rate variability features

**2. Imputation:** Substitute missing values with median imputation from sklearn

**3. Standardization:** Scale the features using the standard scaler from sklearn

**4. Feature selection:**
- Choose features based on the variance
- Additionally determine feature importance using the LGBM regressor and choose a small percentage of the most relevant features in the end

**5. Prediction:** Use the LGBM classifier to predict the class for each electrocardiogram

**Parameter search:** All relevant parameters have been tuned using 5-fold cross validation and randomized search, as provided by sklearn. To speed up tuning,
intermediate steps were saved to disk.