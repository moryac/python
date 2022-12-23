from connector.pg_connector import get_data
from conf.conf import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from util.util import save_model, load_model
from conf.conf import settings
from model.random_forest import train_random_forest, split
from model.logistic_regression import train_regression_model, split

ditc = {'REG': train_regression_model,
'RANDOM': train_random_forest
}

def init_model(model_name: str) -> None:

    df = get_data(settings.DATA.DATASET)
    X_train, X_test, y_train, y_test = split(df)
    clf = ditc[model_name](X_train, y_train)
    logging.info(f'Accuracy is {clf.score(X_test, y_test)}')
    logging.info(f'Prediction is {predict(X_test, settings.MODEL[model_name])}')

def predict(values, path_to_model):
    clf = load_model(path_to_model)
    return clf.predict(values)