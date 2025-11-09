import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_path, test_path):
        try:
            logging.info("Starting model training process")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df[['Recency', 'Frequency', 'Monetary']]
            y_train = train_df['Monetary']

            X_test = test_df[['Recency', 'Frequency', 'Monetary']]
            y_test = test_df['Monetary']

            rf = RandomForestRegressor(random_state=42)

            param_dist = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            logging.info("Performing RandomizedSearchCV for hyperparameter tuning")
            rf_random = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                verbose=2,
                random_state=42,
                n_jobs=-1
            )
            rf_random.fit(X_train, y_train)

            best_rf = rf_random.best_estimator_
            metrics = evaluate_model(best_rf, X_test, y_test)

            logging.info(f"Model evaluation completed: {metrics}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_rf
            )

            logging.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")

            return {"best_params": rf_random.best_params_, **metrics}

        except Exception as e:
            raise CustomException(e, sys)
