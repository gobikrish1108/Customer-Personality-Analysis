import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataIngestionConfig:
    folder_path: str = os.path.join('notebook', 'data')
    file_name: str = 'marketing_campaign.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_and_save_data(self):
        logging.info('Data Ingestion Method Start')

        try:
            # Read the CSV file
            file_path = os.path.join(self.ingestion_config.folder_path, self.ingestion_config.file_name)
            df = pd.read_csv(file_path)
            logging.info("Dataset Read as a Pandas dataframe")

            # Drop unwanted columns
            df.drop('unnamed: 0', axis=1, inplace=True)

            # Save the dataframe to a pickle file
            output_path = os.path.join('artifacts', 'raw_data.pkl')
            save_object(file_path=output_path, obj=df)
            logging.info('DataFrame values have been successfully saved to a pickle file')

            logging.info('Ingestion of Data is completed')

        except Exception as e:
            logging.error('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)

        return output_path
