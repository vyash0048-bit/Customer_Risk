import os
import pandas as pd
from sklearn.model_selection import train_test_split
from CustomerRiskPrediction.logger import logger
from CustomerRiskPrediction.entity.config_entity import DataTransformationConfig
import sys

# Assume woe_iv.py is at the root of src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from woe_iv import calculate_woe_iv, transform_to_woe

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def perform_transformation(self):
        try:
            logger.info("Reading data from ingestion artifacts")
            df = pd.read_csv(self.config.data_path)

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
            
            logger.info(f"Train size: {train.shape}, Val size: {validation.shape}, Test size: {test.shape}")

            logger.info("Binning numerical variables")
            # Binning Age
            train['age_binned'] = pd.cut(train['age'], bins=[17, 25, 35, 45, 55, 120], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            validation['age_binned'] = pd.cut(validation['age'], bins=[17, 25, 35, 45, 55, 120], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            test['age_binned'] = pd.cut(test['age'], bins=[17, 25, 35, 45, 55, 120], labels=['18-25', '26-35', '36-45', '46-55', '56+'])

            # Using qcut on train for duration, and mapping the bins to val/test
            train['duration_binned'], duration_bins = pd.qcut(train['duration'], q=5, duplicates="drop", retbins=True)
            validation['duration_binned'] = pd.cut(validation['duration'], bins=duration_bins, include_lowest=True)
            test['duration_binned'] = pd.cut(test['duration'], bins=duration_bins, include_lowest=True)

            # Convert intervals to string representation
            train['duration_binned'] = train['duration_binned'].astype(str)
            validation['duration_binned'] = validation['duration_binned'].astype(str)
            test['duration_binned'] = test['duration_binned'].astype(str)

            logger.info("Calculating WOE & IV mapping on Train Data")
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'credit_risk']
            candidate_features = categorical_cols + ['age_binned', 'duration_binned']
            
            all_woe_rules = []
            features_to_transform = []

            for f in candidate_features:
                rules = calculate_woe_iv(train, f, 'credit_risk')
                rules['feature'] = f
                total_iv = rules['IV'].sum()
                
                # Keep variables with IV >= 0.02
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

            # Keep only original non-categorical features and newly created WOE features
            numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'credit_risk']
            woe_cols = [f"{f}_WOE" for f in features_to_transform]
            final_columns = numerical_cols + woe_cols + ['credit_risk']

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
