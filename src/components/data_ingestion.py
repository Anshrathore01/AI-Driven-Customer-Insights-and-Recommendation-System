import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load raw transactions
            raw_path = '/Users/anshrathore/Desktop/AI-Driven-Customer-Insights-and-Recommendation-System/src/notebook/data/data.csv'
            logging.info(f"Reading raw data from {raw_path}")
            raw_df = pd.read_csv(raw_path, encoding='latin1')
            
            # Clean transactions
            raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
            raw_df.dropna(subset=['CustomerID'], inplace=True)
            raw_df['CustomerID'] = raw_df['CustomerID'].astype(int)
            raw_df['Quantity'] = raw_df['Quantity'].fillna(0)
            raw_df['UnitPrice'] = raw_df['UnitPrice'].fillna(0)
            raw_df['TotalAmount'] = raw_df['Quantity'] * raw_df['UnitPrice']
            
            # Define cutoff date for 90 days (3 months) future window
            max_date = raw_df['InvoiceDate'].max()
            cutoff_date = max_date - pd.Timedelta(days=90)
            logging.info(f"Splitting transactions: cutoff date is {cutoff_date}")
            
            # Split transactions into history and future target windows
            df_history = raw_df[raw_df['InvoiceDate'] <= cutoff_date]
            df_future = raw_df[raw_df['InvoiceDate'] > cutoff_date]
            
            # Aggregate historical features
            latest_history_date = cutoff_date + pd.Timedelta(days=1)
            rfm_history = df_history.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (latest_history_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalAmount': 'sum'
            }).reset_index()
            
            rfm_history.rename(columns={
                'InvoiceDate': 'Recency',
                'InvoiceNo': 'Frequency',
                'TotalAmount': 'Monetary'
            }, inplace=True)
            
            # Aggregate future spend (money spent in next 3 months)
            future_spend = df_future.groupby('CustomerID')['TotalAmount'].sum().reset_index()
            future_spend.rename(columns={'TotalAmount': 'Target_Spend'}, inplace=True)
            
            # Merge features and targets
            df = pd.merge(rfm_history, future_spend, on='CustomerID', how='left')
            df['Target_Spend'] = df['Target_Spend'].fillna(0)
            
            # Export regenerated final_rfm.csv for record keeping
            rfm_output_path = '/Users/anshrathore/Desktop/AI-Driven-Customer-Insights-and-Recommendation-System/src/notebook/data/final_rfm.csv'
            df.to_csv(rfm_output_path, index=False)
            logging.info(f"Regenerated final_rfm.csv with columns: {df.columns.tolist()}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

