from connector.pg_connector import get_data
from conf.conf import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from util.util import save_model, load_model
from conf.conf import settings


def split(df:pd.DataFrame) -> pd.DataFrame:

    logging.info("Defining X and y")
    
    # Variables
    X = df.iloc[:, :-1]
    y = df['target']

    logging.info("Spliting dataset")

    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                        y, #dependent variable
                                                    )
    return X_train, X_test, y_train, y_test

def train_regression_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    # Initialize the model
    clf = LogisticRegression()
    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)
    params = {'penalty':['l2', 'none']}
    searcher = GridSearchCV(clf, params)
    searcher.fit(X_train, y_train)
    clf.set_params(**searcher.best_params_)

    save_model(dir=settings.MODEL.REG, model=clf)

    return clf
