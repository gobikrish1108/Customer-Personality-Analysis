import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.preprocessing_obj = self.create_transformation_pipeline()

    def create_transformation_pipeline(self):
        try:
            logging.info('Creating data transformation pipeline')

            cols = ['Age', 'Total_Expenses', 'Customer_Tenure_Month', 'Children', 'Income']

            pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=3))
                ]
            )

            preprocessor = ColumnTransformer([
                ('pipeline', pipeline, cols)
            ])

            logging.info('Data transformation pipeline created successfully')

            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)

    def transform_data(self, input_file_path):
        try:
            logging.info(f'Reading data from file {input_file_path}')

            df = pd.read_pickle(input_file_path)

            # Extract relevant columns
            df = df[['Age', 'Spent', 'customer_for', 'children', 'income']]

            logging.info(f'Data loaded successfully from {input_file_path}')
            logging.info(f'Dataframe Head: \n{df.head().to_string()}')

            logging.info('Applying preprocessing to the dataset')
            transformed_data = self.preprocessing_obj.fit_transform(df)

            logging.info('Saving preprocessing object to file')
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=self.preprocessing_obj)

            logging.info('Preprocessing object saved successfully')

            return transformed_data, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f'Error occurred during data transformation: {e}')
            raise CustomException(e, sys)
