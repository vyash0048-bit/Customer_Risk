import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CustomerRiskPrediction.logger import logger
from CustomerRiskPrediction.entity.config_entity import DataTransformationConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from woe_iv import calculate_woe_iv, transform_to_woe

PROTECTED_ATTRIBUTES = {"personal_status_sex", "foreign_worker"}


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def _qcut_open_ended(train_series: pd.Series, val_series: pd.Series, test_series: pd.Series, q: int = 5):
        """Fit quantile bin edges on training data and extend outer edges to (-inf, +inf)."""
        _, bin_edges = pd.qcut(train_series, q=q, duplicates="drop", retbins=True)
        bin_edges = bin_edges.astype(float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        train_binned = pd.cut(train_series, bins=bin_edges, include_lowest=True).astype(str)
        val_binned = pd.cut(val_series, bins=bin_edges, include_lowest=True).astype(str)
        test_binned = pd.cut(test_series, bins=bin_edges, include_lowest=True).astype(str)
        return train_binned, val_binned, test_binned

    def perform_transformation(self):
        try:
            logger.info("Reading data from ingestion artifacts")
            df = pd.read_csv(self.config.data_path)
            logger.info(f"df.dtypes:\n{df.dtypes}")

            logger.info("Splitting data into train, val, test")
            train, temp = train_test_split(
                df,
                test_size=0.30,
                stratify=df["credit_risk"],
                random_state=42
            )

            validation, test = train_test_split(
                temp,
                test_size=0.50,
                stratify=temp["credit_risk"],
                random_state=42
            )
            train = train.copy()
            validation = validation.copy()
            test = test.copy()

            logger.info(f"Train size: {train.shape}, Val size: {validation.shape}, Test size: {test.shape}")

            logger.info("Binning numerical variables with open-ended outer bounds")
            age_bins = [-np.inf, 25, 35, 45, 55, np.inf]
            age_labels = ['<=25', '26-35', '36-45', '46-55', '56+']
            train['age_binned'] = pd.cut(train['age'], bins=age_bins, labels=age_labels).astype(str)
            validation['age_binned'] = pd.cut(validation['age'], bins=age_bins, labels=age_labels).astype(str)
            test['age_binned'] = pd.cut(test['age'], bins=age_bins, labels=age_labels).astype(str)

            train['duration_binned'], validation['duration_binned'], test['duration_binned'] = self._qcut_open_ended(
                train['duration'], validation['duration'], test['duration'], q=5
            )
            train['credit_amount_binned'], validation['credit_amount_binned'], test['credit_amount_binned'] = self._qcut_open_ended(
                train['credit_amount'], validation['credit_amount'], test['credit_amount'], q=5
            )

            logger.info("Calculating WOE & IV mapping on Train Data")
            categorical_cols = [
                col for col in df.columns
                if df[col].dtype.name in ['object', 'category', 'string', 'str'] and col != 'credit_risk'
            ]
            if self.config.exclude_protected_attributes:
                logger.info(f"Excluding protected attributes: {sorted(PROTECTED_ATTRIBUTES)}")
                categorical_cols = [col for col in categorical_cols if col not in PROTECTED_ATTRIBUTES]

            logger.info(f"Categorical cols identified: {categorical_cols}")
            candidate_features = categorical_cols + ['age_binned', 'duration_binned', 'credit_amount_binned']

            all_woe_rules = []
            features_to_transform = []

            for f in candidate_features:
                rules = calculate_woe_iv(train, f, 'credit_risk')
                rules['feature'] = f
                total_iv = rules['IV'].sum()

                if total_iv >= 0.02:
                    all_woe_rules.append(rules)
                    features_to_transform.append(f)
                    logger.info(f"Feature '{f}' IV: {total_iv:.4f} (Keep)")
                else:
                    logger.info(f"Feature '{f}' IV: {total_iv:.4f} (Remove)")

            woe_rules_df = pd.concat(all_woe_rules, ignore_index=True)

            logger.info("Applying WOE transformation")
            for f in features_to_transform:
                rules = woe_rules_df[woe_rules_df['feature'] == f]
                train = transform_to_woe(train, f, rules)
                validation = transform_to_woe(validation, f, rules)
                test = transform_to_woe(test, f, rules)

            woe_cols = [f"{f}_WOE" for f in features_to_transform]
            final_columns = list(df.columns) + ['age_binned', 'duration_binned', 'credit_amount_binned'] + woe_cols

            train = train[final_columns]
            validation = validation[final_columns]
            test = test[final_columns]

            logger.info(f"Saving transformed datasets and WOE rules to {self.config.root_dir}")
            train.to_csv(self.config.transformed_train_path, index=False)
            validation.to_csv(self.config.transformed_val_path, index=False)
            test.to_csv(self.config.transformed_test_path, index=False)
            woe_rules_df.to_csv(self.config.woe_rules_path, index=False)

            logger.info("Data Transformation completed successfully")

        except Exception as e:
            logger.error(f"Error occurred during data transformation: {e}")
            raise e
