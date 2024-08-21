import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test arrays into input features and target variable')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "linear_regression": LinearRegression(),
                "knn": KNeighborsRegressor(),
                "xgboost": XGBRegressor(),
                "ada_boost": AdaBoostRegressor(),
                "catboost": CatBoostRegressor(),
            }

            params = {
                "decision_tree": {
                    'criterion': ['absolute_error', 'friedman_mse'],
                    'splitter': ['best'],
                    'max_features': ['sqrt'],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "random_forest": {
                    'criterion': ['absolute_error', 'squared_error','friedman_mse'],
                    'max_features': ['sqrt'],
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "gradient_boosting": {
                    'loss': ['ls', 'huber'],
                    'learning_rate': [0.1, 0.01],
                    'subsample': [0.7, 0.9],
                    'criterion': ['friedman_mse'],
                    'max_features': ['sqrt'],
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                "linear_regression": {},
                "xgboost": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                },
                "catboost": {
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [100, 200],
                    'l2_leaf_reg': [3, 5],
                },
                "ada_boost": {
                    'learning_rate': [0.1, 0.5],
                    'n_estimators': [50, 100],
                    'loss': ['linear'],
                    'estimator': [DecisionTreeRegressor(max_depth=1)],
                }
            }



            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score = max([model['test_score'] for model in model_report.values()])
            best_model_name = [model_name for model_name, model in model_report.items() if model['test_score'] == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('Best model score is less than 0.6', sys)

            # Logging best model and its hyperparameters
            logging.info(f'Best model is {best_model_name} with test score {best_model_score}')
            best_params = model_report[best_model_name]['best_params']
            logging.info(f'Best hyperparameters for {best_model_name}: {best_params}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)