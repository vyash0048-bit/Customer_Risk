import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from CustomerRiskPrediction.logger import logger
from CustomerRiskPrediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            logger.info("Reading train data")
            train_data = pd.read_csv(self.config.train_data_path)

            target_column = self.config.target_column
            woe_columns = [col for col in train_data.columns if col.endswith('_WOE')]
            
            logger.info(f"Using WOE features for training: {woe_columns}")
            X_train = train_data[woe_columns]
            y_train = train_data[target_column]

            logger.info(f"Training Logistic Regression with solver={self.config.solver}, max_iter={self.config.max_iter}")
            lr = LogisticRegression(
                solver=self.config.solver,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
            lr.fit(X_train, y_train)

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            logger.info(f"Saving model to {model_path}")
            joblib.dump(lr, model_path)

            logger.info("Model Trainer stage completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Trainer: {e}")
            raise e
