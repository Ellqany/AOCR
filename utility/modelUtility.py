import os
import math
import tensorflow as tf

from . import DataGenerator
from attentionocr import (Vectorizer, Vocabulary, AttentionOCR)


def train_attention_ocr(model: AttentionOCR, vec: Vectorizer, model_name: str, data_directory: str, batch_size: int = 64, epochs: int = 1):
    generator = DataGenerator()

    # create data genrator for trainset
    (trainlength, train_data) = generator.csv_data_source(
        vec, data_directory, 'train.txt', True)

    train_gen = tf.data.Dataset.from_generator(
        train_data, output_types=(tf.float32, tf.float32, tf.float32))

    # create data genrator for validationset
    if os.path.isfile(os.path.join(data_directory, 'validation.txt')):
        (_, validation_data) = generator.csv_data_source(
            vec, data_directory, 'validation.txt')

        validation_gen = tf.data.Dataset.from_generator(
            validation_data, output_types=(tf.float32, tf.float32, tf.float32))
    else:
        validation_gen = None

    # calculate the number of iterations required to validate the model normally every epoch
    validate_every_steps = math.ceil(trainlength / batch_size)

    model.fit_generator(
        train_gen,
        validate_every_steps=validate_every_steps,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_gen,
    )

    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    model.save(model_name)


def test_attention_ocr(model: AttentionOCR, voc: Vocabulary, vec: Vectorizer, visualization: bool, data_directory: str):
    generator = DataGenerator()

    (_, test_data) = generator.csv_data_source(vec, data_directory, 'test.txt')

    # create output folder
    if(visualization):
        os.makedirs('out/Correct', exist_ok=True)
        os.makedirs('out/Wrong', exist_ok=True)

    # carry the correct and false prediction (array of bool)
    predictions = []
    for image, _, decoder_output in test_data():
        txt = voc.one_hot_decode(decoder_output, 12)
        (pred, prop) = model.predict([image])
        confidence = prop[0] * 100
        text = '{}: {:.2f}%'.format(pred[0], confidence)

        if (pred[0] == txt):
            predictions.append(True)
        else:
            predictions.append(False)

        if(visualization):
            model.visualise([image], txt)

        print(txt, 'prediction: ', text)

    # get the accuracy on the test dataset
    accuracy = (len([a for a in predictions if a == True]) /
                len(predictions)) * 100

    text = 'Tested Accuracy: {:.2f}%'.format(accuracy)
    print(text)
