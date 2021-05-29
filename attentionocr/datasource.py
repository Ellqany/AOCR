import os
import random
import logging
import traceback

from functools import partial
from . import Vectorizer


LOG = logging.getLogger(__file__)


def csv_data_source(vectorizer: Vectorizer, directory: str, filename: str, is_training: bool = False, sep: str = ';'):
    '''
        convert the dataset to tensor ie (convert the image to tensor and the responding text in each image to tensor)
            vectorizer: help in transforming the image and text to responed tensor
            directory: the directory where the train, test or validation text file stored in
            filename: the train, test or validation text file which the image path and the corresponding text are recodeing in with speprator
            sep: the seprator char to split eachline with to get the image nad corresponding text
            is_training: is the data for training or not
    '''

    examples = []
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            if sep in line:
                image_file, txt = line.split(sep=sep, maxsplit=1)
                image_file = os.path.abspath(
                    os.path.join(directory, image_file))
                txt = txt.strip()
                if os.path.isfile(image_file):
                    examples.append((txt, image_file))
    return (len(examples), partial(examples_generator, examples=examples, vectorizer=vectorizer, is_training=is_training))


def examples_generator(examples: list, vectorizer: Vectorizer, is_training: bool):
    '''
        convert the dataset to tensor ie (convert the image to tensor and the responding text in each image to tensor)
            examples: list containes the image pass and the corresponding text
            vectorizer: help in transforming the image and text to responed tensor
            is_training: is the data for training or not
    '''

    random.shuffle(examples)
    for text, image_file in examples:
        try:
            image = vectorizer.load_image(image_file)
            decoder_input, decoder_output = vectorizer.transform_text(
                text, is_training)
            yield image, decoder_input, decoder_output
        except Exception as err:
            LOG.warning(err)
            traceback.print_tb(err.__traceback__)
