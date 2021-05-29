import os
import tensorflow as tf

from os import path
from attentionocr import (Vectorizer, AttentionOCR,
                          Vocabulary, csv_data_source)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    image_width = 250
    image_height = 150
    data_directory = 'D:\Python\Data\Images\ANPR\Plate2\Data'
    voc = Vocabulary()
    vec = Vectorizer(
        vocabulary=voc, image_width=image_width, image_height=image_height, max_txt_length=12)

    model = AttentionOCR(
        vocabulary=voc, image_width=image_width, image_height=image_height, max_txt_length=12, lr=0)

    (_, test_data) = csv_data_source(vec, data_directory, "test.txt")

    model.load(path.join('D:\Python\Data\Images\ANPR\Plate2', 'Models/ocr10.h5'))
    os.makedirs('out/Correct', exist_ok=True)
    os.makedirs('out/Wrong', exist_ok=True)

    for image, decoder_input, decoder_output in test_data():
        txt = voc.one_hot_decode(decoder_output, 12)
        (pred, prop) = model.predict([image])
        text = "{}: {:.4f}".format(pred[0], prop[0])
        model.visualise([image], txt)
        print(txt, "prediction: ", text)
