import os
import json
import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from CustomerRiskPrediction.entity.config_entity import ModelTrainerConfig
from CustomerRiskPrediction.logger import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
        """Compute Variance Inflation Factor for each feature column."""
        X_const = add_constant(X)
        vif = pd.DataFrame()
        vif["Feature"] = X_const.columns
        vif["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        return vif[vif["Feature"] != "const"]

    @staticmethod
    def compute_prior_odds_shift(pi_true: float, pi_resampled: float) -> float:
        """Return Bayes log-odds intercept offset restoring true class priors after resampling."""
        return float(np.log(pi_true / (1.0 - pi_true)) - np.log(pi_resampled / (1.0 - pi_resampled)))

    def train(self):
        try:
            logger.info("Reading train and validation data")
            train_data = pd.read_csv(self.config.train_data_path)
            val_data = pd.read_csv(self.config.val_data_path)

            target_column = self.config.target_column
            woe_columns = [col for col in train_data.columns if col.endswith('_WOE')]

            logger.info(f"Using WOE features for training: {woe_columns}")
            X_train = train_data[woe_columns]
            y_train = train_data[target_column]

            logger.info("Performing Feature Selection (Multicollinearity - VIF)")
            max_vif = 10.0
            while max_vif > 5.0:
                vif_df = self.calculate_vif(X_train)
                max_vif = vif_df["VIF"].max()
                if max_vif > 5.0:
                    feature_to_drop = vif_df.sort_values(by="VIF", ascending=False).iloc[0]["Feature"]
                    logger.info(f"Dropping feature {feature_to_drop} with VIF {max_vif:.2f}")
                    X_train = X_train.drop(columns=[feature_to_drop])
                else:
                    logger.info("All remaining features have VIF <= 5.0")

            final_features = X_train.columns.tolist()
            logger.info(f"Final selected features ({len(final_features)}): {final_features}")

            X_val = val_data[final_features]
            y_val = val_data[target_column]

            logger.info(
                f"Performing Optuna Bayesian Optimization ({self.config.n_trials} trials) "
                f"with SMOTE inside 5-fold Stratified CV (solver={self.config.solver})..."
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)

            def lr_objective(trial):
                c = trial.suggest_float('C', 1e-4, 1e2, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

                pipe = ImbPipeline([
                    ("smote", SMOTE(random_state=self.config.random_state)),
                    ("clf", LogisticRegression(
                        solver=self.config.solver,
                        C=c,
                        l1_ratio=l1_ratio,
                        class_weight=None,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state
                    ))
                ])

                return cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1).mean()

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
            lr_study = optuna.create_study(direction='maximize', sampler=sampler)
            lr_study.optimize(lr_objective, n_trials=self.config.n_trials)

            logger.info(f"Best LR parameters found: {lr_study.best_params}")
            logger.info(f"Best LR cross-validation ROC-AUC: {lr_study.best_value:.4f}")

            logger.info("Applying SMOTE to full training set and fitting final ElasticNet model...")
            smote = SMOTE(random_state=self.config.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"Resampled train data shape: {X_train_resampled.shape}")

            lr = LogisticRegression(
                solver=self.config.solver,
                C=lr_study.best_params['C'],
                l1_ratio=lr_study.best_params['l1_ratio'],
                class_weight=None,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
            lr.fit(X_train_resampled, y_train_resampled)

            pi_true = float(y_train.mean())
            pi_smote = float(y_train_resampled.mean())
            prior_shift = self.compute_prior_odds_shift(pi_true, pi_smote)
            raw_intercept = float(lr.intercept_[0])
            lr.intercept_ = lr.intercept_ + prior_shift
            logger.info(
                f"Bayes prior-odds intercept recalibration: pi_true={pi_true:.4f}, "
                f"pi_smote={pi_smote:.4f}, shift={prior_shift:.4f}, "
                f"intercept {raw_intercept:.4f} -> {float(lr.intercept_[0]):.4f}"
            )

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            logger.info(f"Saving calibrated Logistic Regression model to {model_path}")
            joblib.dump(lr, model_path)

            # Tune F2-optimal probability threshold on validation set (val.csv) using calibrated probabilities
            y_val_probs = lr.predict_proba(X_val)[:, 1]
            best_threshold = 0.5
            best_f2 = -1.0
            for t in np.linspace(0.10, 0.90, 81):
                preds = (y_val_probs >= t).astype(int)
                f2 = fbeta_score(y_val, preds, beta=2.0, zero_division=0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_threshold = float(round(t, 4))

            logger.info(
                f"Optimal probability threshold on validation set (max F2={best_f2:.4f}): {best_threshold:.4f}"
            )
            threshold_path = os.path.join(self.config.root_dir, "lr_threshold.json")
            with open(threshold_path, 'w') as f:
                json.dump(
                    {
                        'lr_threshold': float(best_threshold),
                        'val_f2_score': float(best_f2),
                        'pi_true': pi_true,
                        'pi_smote': pi_smote,
                        'prior_odds_shift': prior_shift
                    },
                    f,
                    indent=2
                )

            logger.info("Model Trainer stage completed successfully")

        except Exception as e:
            logger.error(f"Error in Model Trainer: {e}")
            raise e
