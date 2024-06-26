import os
import requests
import zipfile
from tqdm import tqdm

from ImageCaptionGenerator import logger
from ImageCaptionGenerator.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config:DataIngestionConfig) -> None:
        self.config = config
    
    def download_and_extract(self,url, dest_path):
        """
        Function to download data and extract zip files

        Args : 
        url : url to resource to be downloaded
        dest_path : path to save file

        """
        try :
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            zip_path = os.path.join(dest_path, 'temp.zip')

            # Streaming download with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte

            logger.info(f"Downloading file to {dest_path} of size {total_size / (1024 * 1024):.2f} MB")

            with open(zip_path, 'wb') as f:
                for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB', unit_scale=True):
                    f.write(data)

            logger.info(f"Download zip file at {dest_path} Completed")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
            os.remove(zip_path)

            logger.info(f"Unziped to {dest_path}")

        except Exception as e:
            raise(e)

    def get_data(self):
        root_dir = self.config.dataset_path

        # download train images
        train_path = os.path.join(root_dir,'train')
        self.download_and_extract(self.config.train_data_url, train_path)

        # download validation images
        val_path = os.path.join(root_dir,'val')
        self.download_and_extract(self.config.validation_data_url, val_path)

        # download annotations
        annot_path = os.path.join(root_dir,'annotations')
        self.download_and_extract(self.config.data_anotations_url, annot_path)

        #download annotations of test images
        annot_test_path = os.path.join(root_dir,'annot_test')
        self.download_and_extract(self.config.data_test_anotations_url, annot_test_path)