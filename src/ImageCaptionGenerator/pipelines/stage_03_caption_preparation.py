from ImageCaptionGenerator.components.prepare_caption import PrepareCaption
from ImageCaptionGenerator.config.configuration import ConfugarationManager

from ImageCaptionGenerator import logger

STAGE_NAME = "CAPTIOIN PREPARATION"

class CaptionPreparationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfugarationManager()
        caption_preparation_config = config.get_prepare_captions_config()
        captioin_preparation = PrepareCaption(config=caption_preparation_config)
        captioin_preparation.prepare_train_sequences()


if __name__=="__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CaptionPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
