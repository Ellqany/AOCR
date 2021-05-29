import tensorflow as tf
import argparse
import os

from attentionocr import (Vectorizer, AttentionOCR, Vocabulary)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--image_width", type=int, default=320, required=False)
    # parser.add_argument("--image_height", type=int, default=32, required=False)
    parser.add_argument("--max_txt_length", type=int,
                        default=42, required=False)
    parser.add_argument("--tf_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--learining_rate", type=float,
                        default=0, required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(
        vocabulary=voc, image_width=args.image_width, max_txt_length=args.max_txt_length)

    model = AttentionOCR(
        vocabulary=voc, image_width=args.image_width, max_txt_length=args.max_txt_length, lr=args.learining_rate)

    model.load(args.tf_model)

    # Conver to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model._training_model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(args.model_name), exist_ok=True)

    # Save the model.
    with open(args.model_name, 'wb') as f:
        f.write(tflite_model)
