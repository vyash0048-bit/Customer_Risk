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
            model_lr = joblib.load(self.config.model_path)
            
            # Paths for xgb
            xgb_model_path = self.config.model_path.replace("model.joblib", "xgb_model.joblib")
            model_xgb = joblib.load(xgb_model_path)

            target_column = self.config.target_column
            
            if hasattr(model_lr, 'feature_names_in_'):
                woe_columns = list(model_lr.feature_names_in_)
            else:
                woe_columns = [col for col in test_data.columns if col.endswith('_WOE')]
            
            logger.info(f"Using WOE features for evaluation: {woe_columns}")
            X_test = test_data[woe_columns]
            y_test = test_data[target_column]

            logger.info("Predicting on test data with Logistic Regression")
            y_pred_lr = model_lr.predict(X_test)
            y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

            logger.info("Calculating LR metrics")
            metrics_lr = {
                "ROC_AUC": float(roc_auc_score(y_test, y_pred_proba_lr)),
                "Gini": float(2 * roc_auc_score(y_test, y_pred_proba_lr) - 1),
                "PR_AUC": float(average_precision_score(y_test, y_pred_proba_lr)),
                "Log_Loss": float(log_loss(y_test, y_pred_proba_lr)),
                "Brier_Score": float(brier_score_loss(y_test, y_pred_proba_lr)),
                "Precision": float(precision_score(y_test, y_pred_lr, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred_lr, zero_division=0)),
                "F1_Score": float(f1_score(y_test, y_pred_lr, zero_division=0)),
                "Confusion_Matrix": {
                    "TN": int(confusion_matrix(y_test, y_pred_lr)[0, 0]),
                    "FP": int(confusion_matrix(y_test, y_pred_lr)[0, 1]),
                    "FN": int(confusion_matrix(y_test, y_pred_lr)[1, 0]),
                    "TP": int(confusion_matrix(y_test, y_pred_lr)[1, 1])
                }
            }
            logger.info(f"LR Metrics evaluated: {metrics_lr}")
            save_json(path=Path(self.config.metric_file_name), data=metrics_lr)
            
            # --- XGBoost Evaluation ---
            logger.info("Predicting on test data with XGBoost")
            y_pred_xgb = model_xgb.predict(X_test)
            y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

            logger.info("Calculating XGBoost metrics")
            metrics_xgb = {
                "ROC_AUC": float(roc_auc_score(y_test, y_pred_proba_xgb)),
                "Gini": float(2 * roc_auc_score(y_test, y_pred_proba_xgb) - 1),
                "PR_AUC": float(average_precision_score(y_test, y_pred_proba_xgb)),
                "Log_Loss": float(log_loss(y_test, y_pred_proba_xgb)),
                "Brier_Score": float(brier_score_loss(y_test, y_pred_proba_xgb)),
                "Precision": float(precision_score(y_test, y_pred_xgb, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred_xgb, zero_division=0)),
                "F1_Score": float(f1_score(y_test, y_pred_xgb, zero_division=0)),
                "Confusion_Matrix": {
                    "TN": int(confusion_matrix(y_test, y_pred_xgb)[0, 0]),
                    "FP": int(confusion_matrix(y_test, y_pred_xgb)[0, 1]),
                    "FN": int(confusion_matrix(y_test, y_pred_xgb)[1, 0]),
                    "TP": int(confusion_matrix(y_test, y_pred_xgb)[1, 1])
                }
            }
            logger.info(f"XGBoost Metrics evaluated: {metrics_xgb}")
            xgb_metric_path = str(self.config.metric_file_name).replace("metrics.json", "xgb_metrics.json")
            save_json(path=Path(xgb_metric_path), data=metrics_xgb)

            logger.info("Model evaluation completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            raise e
