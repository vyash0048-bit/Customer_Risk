import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
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

            logger.info("Performing Feature Selection (Multicollinearity - VIF)")
            
            # Function to calculate VIF and drop features > 5 iteratively
            def calculate_vif(X):
                X_const = add_constant(X)
                vif = pd.DataFrame()
                vif["Feature"] = X_const.columns
                vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
                return vif[vif["Feature"] != "const"]

            # Iteratively drop features with VIF > 5
            max_vif = 10
            while max_vif > 5.0:
                vif_df = calculate_vif(X_train)
                max_vif = vif_df["VIF"].max()
                if max_vif > 5.0:
                    feature_to_drop = vif_df.sort_values(by="VIF", ascending=False).iloc[0]["Feature"]
                    logger.info(f"Dropping feature {feature_to_drop} with VIF {max_vif:.2f}")
                    X_train = X_train.drop(columns=[feature_to_drop])
                else:
                    logger.info("All remaining features have VIF <= 5.0")
            
            final_features = X_train.columns.tolist()
            logger.info(f"Final selected features ({len(final_features)}): {final_features}")

            logger.info(f"Performing GridSearchCV for Logistic Regression...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            }
            
            base_lr = LogisticRegression(
                solver=self.config.solver,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
            
            grid_search = GridSearchCV(
                estimator=base_lr,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
            
            # The best estimator is refitted automatically
            lr = grid_search.best_estimator_

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            logger.info(f"Saving model to {model_path}")
            joblib.dump(lr, model_path)

            logger.info("Model Trainer stage completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Trainer: {e}")
            raise e
