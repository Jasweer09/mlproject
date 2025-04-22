import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
#used to create the class varaibles
from dataclasses import dataclass

'''dataclass is advantage, it helps to provide a constructor, string representation
and comparision with stating in the class. It gonna provide to us'''

'''
    But my recommondation is use this when you have only variable to class
    If you have methods along with the variables try to do it your own using __init__, __str__ like this
'''
@dataclass
class DataIngestionConfig:
    #These are the file paths that dataingestion will required
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        #for reading the data from the databases
        logging.info("Entered in to the data ingestion method/component")
        try:

            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train Test Split intiated')
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index =False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index =False, header = True)
            logging.info('Ingestion of the data is completed')

            return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

