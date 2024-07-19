import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e,sys)

def calculate_scores(train_array, num_clusters):
    # Apply KMeans clustering to the training data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(train_array)

    # Compute the silhouette score for the clusters
    silhouette_avg = silhouette_score(train_array, clusters)

    # Determine the size of each cluster
    cluster_sizes = [np.sum(clusters == i) for i in range(num_clusters)]

    # Calculate the Soliot score as the ratio of the smallest to the largest cluster size
    soliot_score = min(cluster_sizes) / max(cluster_sizes)

    return silhouette_avg, soliot_score

class DataProcessor:
    def __init__(self, db_url=None):
        self.db_url = db_url
    
    def save_dataframe(self, df, file_name):
        try:
            file_path = os.path.join('notebook', 'data', f'{file_name}.csv')
            df.to_csv(file_path, index=False)
            logging.info(f"DataFrame successfully saved as '{file_path}'")
        except Exception as e:
            raise CustomException(e, sys)
            logging.error(f"An error occurred: {str(e)}")

    def load_dataframe(self, file_name):
        try:
            file_path = os.path.join('notebook', 'data', f'{file_name}.csv')
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)
            logging.error(f"An error occurred: {str(e)}")
            return None  # Return None to indicate an error

    def close(self):
        try:
            logging.info("Data processing completed.")
        except Exception as e:
            raise CustomException(e, sys)
            logging.error(f"An error occurred: {str(e)}")
