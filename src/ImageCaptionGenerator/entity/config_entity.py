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

@dataclass(frozen=True)
class PrepareCaptionConfig:
    train_annotations_path : Path
    param_path : Path
    dest_path : Path
    num_words : int