import os
import requests
import zipfile
from tqdm import tqdm

from ImageCaptionGenerator import logger
from ImageCaptionGenerator.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
    
    def download_file(self, url, dest_path):
        """
        Function to download a file from a URL

        Args:
        url: URL to resource to be downloaded
        dest_path: Path to save the downloaded file
        """
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        file_path = os.path.join(dest_path, 'temp.zip')

        # Streaming download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        logger.info(f"Downloading file to {file_path} of size {total_size / (1024 * 1024):.2f} MB")

        with open(file_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB', unit_scale=True):
                f.write(data)

        logger.info(f"Download of file at {file_path} completed")
        return file_path

    def extract_file(self, zip_path, dest_path):
        """
        Function to extract a zip file

        Args:
        zip_path: Path to the zip file to be extracted
        dest_path: Path to extract the zip file contents
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    try:
                        zip_ref.extract(member, dest_path)
                    except zipfile.BadZipFile as e:
                        logger.error(f"Corrupted file {member} in zip archive {zip_path}: {e}")
            os.remove(zip_path)
            logger.info(f"Unzipped to {dest_path}")
        except zipfile.BadZipFile as e:
            logger.error(f"Bad zip file {zip_path}: {e}")
            raise

    def download_and_extract(self, url, dest_path):
        """
        Function to download data and extract zip files

        Args:
        url: URL to resource to be downloaded
        dest_path: Path to save the file and extract its contents
        """
        zip_path = self.download_file(url, dest_path)
        self.extract_file(zip_path, dest_path)

    def get_data(self):
        root_dir = self.config.dataset_path

        # Download and extract train images
        train_path = os.path.join(root_dir, 'train')
        self.download_and_extract(self.config.train_data_url, train_path)

        # Download and extract validation images
        val_path = os.path.join(root_dir, 'val')
        self.download_and_extract(self.config.validation_data_url, val_path)

        # Download and extract annotations
        annot_path = os.path.join(root_dir, 'annotations')
        self.download_and_extract(self.config.data_anotations_url, annot_path)