import os 
import sys 

import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    '''
    this function is responsible for saving object
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''
    this function is responsible for evaluating models
    '''
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = {'train_score': train_score, 'test_score': test_score}
        return model_report

    except Exception as e:
        raise CustomException(e, sys)