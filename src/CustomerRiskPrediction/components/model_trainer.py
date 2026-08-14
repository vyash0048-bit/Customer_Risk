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

            logger.info("Applying SMOTE for handling class imbalance...")
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"Resampled train data shape: {X_train_resampled.shape}")

            logger.info(f"Performing Optuna Bayesian Optimization for Logistic Regression (ElasticNet)...")
            
            import optuna
            from sklearn.model_selection import cross_val_score
            
            def lr_objective(trial):
                # Search space for Logistic Regression
                c = trial.suggest_float('C', 1e-4, 1e2, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
                
                model = LogisticRegression(
                    solver='saga',
                    C=c,
                    l1_ratio=l1_ratio,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=42,
                    n_jobs=1
                )
                
                score = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', n_jobs=1).mean()
                return score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            lr_study = optuna.create_study(direction='maximize')
            lr_study.optimize(lr_objective, n_trials=50)
            
            logger.info(f"Best LR parameters found: {lr_study.best_params}")
            logger.info(f"Best LR cross-validation ROC-AUC: {lr_study.best_value:.4f}")
            
            lr = LogisticRegression(
                solver='saga',
                C=lr_study.best_params['C'],
                l1_ratio=lr_study.best_params['l1_ratio'],
                class_weight=lr_study.best_params['class_weight'],
                max_iter=1000,
                random_state=42
            )
            lr.fit(X_train_resampled, y_train_resampled)
            
            model_path = os.path.join(self.config.root_dir, "model.joblib")
            logger.info(f"Saving Logistic Regression model to {model_path}")
            joblib.dump(lr, model_path)
            
            # Calculate optimal probability threshold on resampled train data
            from sklearn.metrics import fbeta_score
            y_train_probs = lr.predict_proba(X_train_resampled)[:, 1]
            best_threshold = 0.5
            best_f2 = 0
            for t in np.linspace(0.1, 0.9, 81):
                preds = (y_train_probs >= t).astype(int)
                f2 = fbeta_score(y_train_resampled, preds, beta=2.0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_threshold = t
            
            logger.info(f"Optimal probability threshold for Logistic Regression (max F2): {best_threshold:.4f}")
            threshold_path = os.path.join(self.config.root_dir, "lr_threshold.json")
            import json
            with open(threshold_path, 'w') as f:
                json.dump({'lr_threshold': float(best_threshold)}, f)
            
            logger.info("Model Trainer stage completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Trainer: {e}")
            raise e
