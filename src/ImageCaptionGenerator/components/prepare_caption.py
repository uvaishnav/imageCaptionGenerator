from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
import joblib
import yaml
import os

from ImageCaptionGenerator import logger
from ImageCaptionGenerator.entity.config_entity import PrepareCaptionConfig

class PrepareCaption:
    def __init__(self, config:PrepareCaptionConfig) -> None:
        self.config = config

    def get_captions(self, captions_path):
        """
        Prepare Captions for each image along with their image ids
        """
        # Load the COCO annotations
        with open(captions_path, 'r') as f:
            annotations = json.load(f)
        
        # Prepare the Captions
        captions = []
        images = []

        for annot in annotations['annotations']:
            captions.append(annot['caption'])
            images.append(annot['image_id'])

        logger.info("Captions fetched")
        return captions, images


    def get_tokenizer(self):
        """
        Tokenizes the Captions and Save it.
        """
        # Tokenize the captions
        tokenizer = Tokenizer(num_words=self.config.num_words, oov_token='<unk>')
        captions, _ = self.get_captions(captions_path=self.config.train_annotations_path)

        logger.info("Preparing Tokenizer on train Captions")
        tokenizer.fit_on_texts(captions)
        logger.info("Tokenizer Prepared")

        # Saving Tokenizer For further use
        tokenizer_path = os.path.join(self.config.dest_path, 'tokenizer.pkl')
        joblib.dump(tokenizer, tokenizer_path)
        logger.info("Tokenizer Saved")

        return tokenizer
    
    def get_sequences(self, tokenizer, captions_path):
        """
        Convert Captions into Sequences 

        Args :- captions path to convert to sequences
        Returns :- padded sequences.
        """

        # Get Captions
        captions, _ = self.get_captions(captions_path=captions_path)

        # Convert captions to sequences
        sequences = tokenizer.texts_to_sequences(captions)
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        logger.info("Sequences Generated from Captions")

        return padded_sequences, max_length
    
    def prepare_train_sequences(self):
        """
        Prepare The training Sequences and save them along with the parameters like Vocabulary size and max length
        """
        # Get tokenizer
        tokenizer = self.get_tokenizer()

        # Get vocab_size
        vocab_size = len(tokenizer.word_index) + 1

        # Get Padded Sequences and Max Length
        logger.info("Get Sequences from train Captions")
        padded_sequences, max_length = self.get_sequences(tokenizer=tokenizer, captions_path=self.config.train_annotations_path)

        # Save padded sequences
        train_sequences_path = os.path.join(self.config.dest_path, 'train_padded_sequences.pkl')
        joblib.dump(padded_sequences, train_sequences_path)
        logger.info("Saved Padded Sequences from train Captions")

        # Save Parameters
        # Read existing parameters from params.yaml
        try:
            with open(self.config.param_path, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            params = {}

        # Update parameters with new values
        params.update({
            'vocab_size': vocab_size,
            'max_length': max_length
        })

        # Save updated parameters to params.yaml
        with open('params.yaml', 'w') as f:
            yaml.dump(params, f)
        
        logger.info("Params Updated")









        






