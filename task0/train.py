from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas
import numpy

class MyModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return numpy.mean(X, axis=1)

def write_results(model):
    y_pred = model.predict(get_test_data())
    df = pandas.DataFrame(y_pred, columns=["y"])
    df.index += 10000
    df.to_csv("submission.csv", index_label="Id")

def test(X, y, model):
    y_pred = model.predict(X)
    return mean_squared_error(y_pred, y)

def train(X, y, model):
    model.fit(X, y)
    return model

def get_test_data():
    data = pandas.read_csv("test.csv")
    X = data.loc[:, "x1":].to_numpy()
    return X

def get_eval_data():
    data = pandas.read_csv("train.csv")
    y = data["y"].to_numpy()
    X = data.loc[:, "x1":].to_numpy()
    return X, y

def get_train_data():
    X, y = get_eval_data()
    return train_test_split(X, y, test_size=0.1)

def test_models():
    X_train, X_test, y_train, y_test = get_train_data()
    models = [LinearRegression(), Lasso(), Ridge(), ElasticNet(), SGDRegressor(), RANSACRegressor(), RandomForestRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), BaggingRegressor(), HistGradientBoostingRegressor(), MyModel()]
    for model in models:
        print(type(model).__name__)
        train(X_train, y_train, model)
        error = test(X_test, y_test, model)
        print(error)


def main():
    # test_models()
    model = train(*get_eval_data(), MyModel())
    write_results(model)

if __name__ == "__main__":
    main()