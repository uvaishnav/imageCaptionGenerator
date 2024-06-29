from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_path : Path
    train_data_url : str
    validation_data_url : str
    data_anotations_url : str

@dataclass(frozen=True)
class FeatureExtractionConfig:
    train_data_path : Path
    val_data_path : Path
    features_path : Path
    