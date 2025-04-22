
import os
import sys

from dataclasses import dataclass
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
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_training(self, train_arr, test_arr):
        try:
            #splitting the train and test array data
            X_train, y_train, X_test, y_test =(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("splitted the train and test array data")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regresson": LinearRegression(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoosting Classifier": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    # 'max_depth': [None, 10, 20],
                    # 'min_samples_split':[2, 5]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [None, 5, 10, 20, 30],
                    # 'min_samples_split':[2, 5, 10],
                    # 'min_samples_leaf':[1, 2, 4],
                    # 'max_features':[None, 'sqrt', 'log2']
                },
                "Gradient Boosting":{
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    # 'max_depth': [3, 5, 10],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'max_features': ['sqrt', 'log2'],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "Linear Regresson": {
                    'fit_intercept': [True, False]
                },
                "K-Neighbours Classifier": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'leaf_size': [20, 30, 40],
                    # 'p': [1, 2]
                },
                "XGBClassifier": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                },
                "CatBoosting Classifier": {
                    'depth':[6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoosting Classifier":{
                    'learning_rate':[.1, .01, 0.5, 0.1],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                }

            }


            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train= y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                param = params)
            
            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name from the dictrionaries
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model

            )
            print(model_report)
            print(best_model_name)
            predicted = best_model.predict(X_test)
            pm_r2_score = r2_score(y_test, predicted)
            return pm_r2_score

        except Exception as e:
            raise CustomException(e, sys)