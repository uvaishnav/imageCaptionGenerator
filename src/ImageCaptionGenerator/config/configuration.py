from ImageCaptionGenerator.constants import *
from ImageCaptionGenerator.utils.common import read_yaml
from ImageCaptionGenerator.entity.config_entity import (
    DataIngestionConfig,
    FeatureExtractionConfig
)

class ConfugarationManager:
    def __init__(
            self,
            config_file_path = CONFIG_FILE_PATH,
            params_file_path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        data_ingestion_config = DataIngestionConfig(
            dataset_path = config.dataset_path,
            train_data_url = config.train_data_url,
            validation_data_url = config.validation_data_url,
            data_anotations_url = config.data_anotations_url
        )

        return data_ingestion_config
    
    def get_feature_extractor_config(self)-> FeatureExtractionConfig:
        config = self.config.feature_extraction
        feature_extractor_config = FeatureExtractionConfig(
            train_data_path = config.train_data_path,
            val_data_path = config.val_data_path,
            features_path = config.features_path
        )

        return feature_extractor_config

        