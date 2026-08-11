import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import urllib.parse
from CustomerRiskPrediction.logger import logger
from CustomerRiskPrediction.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            load_dotenv()
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")

            if not all([db_host, db_port, db_name, db_user, db_password]):
                raise ValueError("Database credentials missing in .env file.")

            # URL encode the password to handle special characters like '@'
            encoded_password = urllib.parse.quote_plus(db_password)

            # Load data from source file
            logger.info(f"Loading data from {self.config.source_data_file}")
            columns = [
                "checking_account", "duration", "credit_history", "purpose",
                "credit_amount", "savings_account", "employment", "installment_rate",
                "personal_status_sex", "other_debtors", "residence_since", "property",
                "age", "other_installment_plans", "housing", "existing_credits",
                "job", "dependents", "telephone", "foreign_worker", "credit_risk"
            ]
            df = pd.read_csv(
                self.config.source_data_file,
                sep=r"\s+",
                header=None,
                names=columns
            )
            logger.info(f"Data loaded successfully. Shape: {df.shape}")

            # Convert Target (1 = Good -> 0, 2 = Bad -> 1)
            df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})
            logger.info(f"Target distribution:\n{df['credit_risk'].value_counts()}")

            # Create connection string
            connection_string = (
                f"postgresql+psycopg2://"
                f"{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
            )

            logger.info("Connecting to PostgreSQL database")
            engine = create_engine(connection_string)
            
            # Load Data into PostgreSQL
            schema, table = self.config.table_name.split('.') if '.' in self.config.table_name else ('public', self.config.table_name)
            
            # Create schema if it doesn't exist
            if schema != 'public':
                with engine.connect() as conn:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                    conn.commit()
                    
            logger.info(f"Loading data into PostgreSQL table {schema}.{table}")
            df.to_sql(
                name=table,
                con=engine,
                schema=schema,
                if_exists="append",
                index=False
            )
            logger.info("Data successfully loaded into PostgreSQL.")

            # Save data locally
            logger.info(f"Saving data to {self.config.local_data_file}")
            df.to_csv(self.config.local_data_file, index=False)
            logger.info(f"Data successfully saved at {self.config.local_data_file}")
            
        except Exception as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            raise e
