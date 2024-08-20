import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTrasformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasformationConfig()

    def get_data_transformer_object(self):

        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race/ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numerical column encoder completed')
            logging.info('Categorical column encoder completed')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        this function is responsible for initiating data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('test and train in df format')
            logging.info('preprocessor object created')
            
            preprocessor = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('input and target feature separated')

            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info('saved preprocessor object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        
        