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

            logger.info(f"Performing GridSearchCV for Logistic Regression...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]
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
                n_jobs=1
            )
            
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
            
            # The best estimator is refitted automatically
            lr = grid_search.best_estimator_

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            logger.info(f"Saving Logistic Regression model to {model_path}")
            joblib.dump(lr, model_path)
            
            # --- Train Advanced Challenger Model (CatBoost with Optuna) ---
            logger.info("Training Advanced Challenger Model (CatBoost) on raw features...")
            import catboost as cb
            import optuna

            # Use the raw features for CatBoost instead of WOE features
            raw_features = [col for col in train_data.columns if not col.endswith('_WOE') and col != target_column]
            X_train_raw = train_data[raw_features].copy()
            
            # Fill missing values for CatBoost
            for col in X_train_raw.columns:
                if X_train_raw[col].dtype.name in ['object', 'category']:
                    X_train_raw[col] = X_train_raw[col].fillna("Unknown").astype(str)
                else:
                    X_train_raw[col] = X_train_raw[col].fillna(0)

            cat_features = [col for col in X_train_raw.columns if X_train_raw[col].dtype.name in ['object', 'category', 'str']]

            def objective(trial):
                params = {
                    'loss_function': 'Logloss',
                    'iterations': trial.suggest_int('iterations', 50, 200),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'None']),
                    'verbose': 0,
                    'cat_features': cat_features
                }
                
                # Cross-validation
                cv_data = cb.cv(
                    cb.Pool(X_train_raw, y_train, cat_features=cat_features),
                    params,
                    fold_count=3,
                    return_models=False,
                    verbose=0
                )
                
                return cv_data['test-Logloss-mean'].iloc[-1]

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)

            logger.info(f"CatBoost Best Optuna params: {study.best_params}")

            best_cb_params = study.best_params
            best_cb_params['cat_features'] = cat_features
            best_cb_params['verbose'] = 0
            
            cb_model = cb.CatBoostClassifier(**best_cb_params)
            cb_model.fit(X_train_raw, y_train)

            cb_model_path = os.path.join(self.config.root_dir, "cb_model.joblib")
            logger.info(f"Saving CatBoost model to {cb_model_path}")
            joblib.dump(cb_model, cb_model_path)

            logger.info("Model Trainer stage completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Trainer: {e}")
            raise e
