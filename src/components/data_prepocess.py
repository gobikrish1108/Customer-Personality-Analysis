import pandas as pd
import numpy as np
from src.exception import CustomException
import os
from src.logger import logging
import sys
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class DataProcessor:
    def __init__(self):
        pass

    def clean_and_reduce_dimensionality(self, df):
        try:
            logging.info('Starting data cleaning process')

            # Process df data
            df = df.dropna()

            # Assuming the date strings are in the format '%d-%m-%Y'
            df['dt_customer'] = pd.to_datetime(df['dt_customer'], format='%d-%m-%Y')

            # Calculate the 'customer_for' column
            df['customer_for'] = 12.0 * (2015 - df['dt_customer'].dt.year) + (1 - df['dt_customer'].dt.month)

            current_year = datetime.now().year
            df['Age'] = current_year - df['year_birth']
            df['Spent'] = df[['mntwines', 'mntfruits', 'mntmeatproducts', 'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']].sum(axis=1)
            df['living_with'] = df['marital_status'].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Single", "Widow": "Single", "YOLO": "Single", "Divorced": "Single", "Single": "Single"})
            df['children'] = df['kidhome'] + df['teenhome']
            df['Family_Size'] = df['living_with'].replace({"Single": 1, "Partner": 2,"Alone":1})
            df['Is_Parent'] = np.where(df['children'] > 0, 1, 0)
            df['education'] = df['education'].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
            df['customer_for'] = pd.to_numeric(df['customer_for'], errors="coerce")
            df.loc[(df['Age'] >= 13) & (df['Age'] <= 19), 'AgeGroup'] = 'Teen'
            df.loc[(df['Age'] >= 20) & (df['Age']<= 39), 'AgeGroup'] = 'Adult'
            df.loc[(df['Age'] >= 40) & (df['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
            df.loc[(df['Age'] >= 60), 'AgeGroup'] = 'Senior Adult'
            df.rename(columns={
                "mntwines": "Wines",
                "mntfruits": "Fruits",
                "mntmeatproducts": "Meat",
                "mntfishproducts": "Fish",
                "mntsweetproducts": "Sweets",
                "mntgoldprods": "Gold"
            }, inplace=True)

            df = df[df.Age < 100]
            df = df[df.income < 120000]

            logging.info('Data cleaning completed')

            # Store the intermediate cleaned dataframe (optional)
            cleaned_df = df.copy()

            # Drop unnecessary columns
            to_drop = ["marital_status", "dt_customer", "z_costcontact", "z_revenue", "year_birth", "id", "AgeGroup", "living_with"]
            df.drop(to_drop, axis=1, inplace=True)

            df.Is_Parent = pd.to_numeric(df.Is_Parent, errors='coerce')

            # Get list of categorical variables
            s = (df.dtypes == 'object')
            object_cols = list(s[s].index)

            le = LabelEncoder()
            for col in object_cols:
                df[col] = le.fit_transform(df[col])

            logging.info("All features are now numerical")

            return df, cleaned_df

        except Exception as e:
            logging.error(f'Error occurred during data processing: {e}')
            raise CustomException(e, sys)
