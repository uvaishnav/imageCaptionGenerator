import tensorflow as tf 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from tqdm import tqdm
import joblib

from ImageCaptionGenerator import logger
from ImageCaptionGenerator.entity.config_entity import FeatureExtractionConfig

class FeatureExtractor:
    def __init__(self, config:FeatureExtractionConfig):
        self.config = config

        # Load pre-trained ResNet50 model 
        self.resent_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def extract_features(self, img_path):
        """
        Extract Features from the image using pre-trained RestNet50 model to be used for LSTM to generate Captions

        Args : Path to image to extract features
        Returns : Feature Vector

        """
        try:
            img = image.load_img(img_path, target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.resent_model.predict(img_array)
            return features
        except OSError as e:
            print(f"Skipping corrupted image: {img_path}, error: {e}")
            return None
    
    def extract_all_features(self, data_path):
        """
        Extract Features from each Image of the dataset and store then in a dictonary.

        """
        features = {}
        for image_name in tqdm(os.listdir(data_path)):
            img_path = os.path.join(data_path, image_name)
            img_feature = self.extract_features(img_path)
            features[image_name] = img_feature
        return features
    
    def get_train_val_features(self):
        """
        Extract features from the image and store them in pickle file for further use.

        """
        if not os.path.exists(self.config.features_path):
            os.makedirs(self.config.features_path)

        logger.info("Extracting Train Features...")
        train_features = self.extract_all_features(self.config.train_data_path)
        logger.info("Training Features Extracted")

        # load train features
        train_features_path = os.path.join(self.config.features_path, 'train_features.pkl')
        joblib.dump(train_features, train_features_path)
        logger.info("Saved train features")
        
        logger.info("Extracting Validation Features...")
        val_features = self.extract_all_features(self.config.val_data_path)
        logger.info("Validation Features Extracted")

        # load validation features
        validation_features_path = os.path.join(self.config.features_path, 'val_features.pkl')
        joblib.dump(val_features, validation_features_path)
        logger.info("Saved Validation Features")

