from ImageCaptionGenerator.components.feature_extractor import FeatureExtractor
from ImageCaptionGenerator.config.configuration import ConfugarationManager

from ImageCaptionGenerator import logger

STAGE_NAME = "FEATURER EXTRACTION"

class FeatureExtractionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfugarationManager()
        feature_extractor_config = config.get_feature_extractor_config()
        feature_extractor = FeatureExtractor(config=feature_extractor_config)
        feature_extractor.get_train_val_features()

if __name__=="__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureExtractionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e