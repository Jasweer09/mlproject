import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #a path to save my models
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    '''
        This function is responsible to perform the transformation on 
        numerical and categorical fetrues
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features =[
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            num_pipeline = Pipeline(
                steps = [
                    #for missing values
                    ("imputer", SimpleImputer(strategy="median")),
                    #scaling
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical transformation pipeline created")
            cat_pipeline =Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical transformation pipeline created")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            logging.info("preprocessor for transformation is created")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the train and test data for transformation")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'math score'
            #dropping the target column from the dataframes of test and train
            input_train_feature_df = train_df.drop(target_column_name, axis =1)
            target_train_feature_df = train_df[target_column_name]

            input_test_feature_df = test_df.drop(target_column_name, axis = 1)
            target_test_feature_df = test_df[target_column_name]
            logging.info('dropped the target column from the dataframes of test and train')

            #Performing the data transformation on both test and train input features
            
            input_train_feature_arr = preprocessor_obj.fit_transform(input_train_feature_df)
            input_test_feature_arr = preprocessor_obj.fit_transform(input_test_feature_df)
            logging.info('Performing the data transformation on both test and train input features')

            train_arr = np.c_[
                input_train_feature_arr, np.array(target_train_feature_df)
            ]

            test_arr = np.c_[
                input_test_feature_arr, np.array(target_test_feature_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            