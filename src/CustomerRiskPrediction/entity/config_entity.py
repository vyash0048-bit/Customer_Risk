from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_file: Path
    local_data_file: Path
    table_name: str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformed_train_path: Path
    transformed_val_path: Path
    transformed_test_path: Path
    woe_rules_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    model_name: str
    target_column: str
    max_iter: int
    random_state: int
    solver: str
