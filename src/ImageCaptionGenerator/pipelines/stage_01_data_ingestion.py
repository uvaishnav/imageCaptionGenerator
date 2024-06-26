from ImageCaptionGenerator.components.data_ingestion import DataIngestion
from ImageCaptionGenerator.config.configuration import ConfugarationManager

from ImageCaptionGenerator import logger

STAGE_NAME = "DATA INGESTION"

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfugarationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.get_data()

if __name__=="__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e