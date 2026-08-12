import os
import pandas as pd
import json
import joblib
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    average_precision_score
)
from CustomerRiskPrediction.logger import logger
from CustomerRiskPrediction.entity.config_entity import ModelEvaluationConfig
from CustomerRiskPrediction.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def save_results(self):
        try:
            logger.info("Reading test data")
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            target_column = self.config.target_column
            woe_columns = [col for col in test_data.columns if col.endswith('_WOE')]
            
            logger.info(f"Using WOE features for evaluation: {woe_columns}")
            X_test = test_data[woe_columns]
            y_test = test_data[target_column]

            logger.info("Predicting on test data")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            logger.info("Calculating metrics")
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            gini = 2 * roc_auc - 1
            pr_auc = average_precision_score(y_test, y_pred_proba)
            loss_log = log_loss(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1])
            }

            metrics = {
                "ROC_AUC": float(roc_auc),
                "Gini": float(gini),
                "PR_AUC": float(pr_auc),
                "Log_Loss": float(loss_log),
                "Brier_Score": float(brier),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1_Score": float(f1),
                "Confusion_Matrix": cm_dict
            }

            logger.info(f"Metrics evaluated: {metrics}")
            
            save_json(path=Path(self.config.metric_file_name), data=metrics)
            logger.info("Model evaluation completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            raise e
