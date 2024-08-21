import os 
import sys 

import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV




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

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    This function is responsible for evaluating models.
    '''
    try:
        model_report = {}
        for model_name, model in models.items():
            # Get the parameters for the specific model
            model_params = params.get(model_name, {})
            
            if model_params:  # If there are parameters to tune
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(model, model_params, cv=5)
                grid_search.fit(X_train, y_train)
                
                # Update the model with the best found parameters
                model.set_params(**grid_search.best_params_)
            else:  # If no parameters to tune, fit the model directly
                model.fit(X_train, y_train)

            # Fit the model on the entire training set
            model.fit(X_train, y_train)

            # Predict using the fitted model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R2 scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            # Store the results and best parameters in the model report
            model_report[model_name] = {
                'train_score': train_score,
                'test_score': test_score,
                'best_params': grid_search.best_params_ if model_params else None
            }
        
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
