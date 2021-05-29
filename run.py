import os
import argparse
import math
import tensorflow as tf

from attentionocr import (Vectorizer, AttentionOCR,
                          Vocabulary, csv_data_source)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--image_width", type=int, default=320, required=False)
    parser.add_argument("--image_height", type=int, default=32, required=False)
    parser.add_argument("--max_txt_length", type=int,
                        default=42, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str,
                        default=None, required=False)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--learining_rate", type=float,
                        default=0, required=False)

    args = parser.parse_args()

    voc = Vocabulary()
    vec = Vectorizer(
        vocabulary=voc, image_width=args.image_width, image_height=args.image_height, max_txt_length=args.max_txt_length)
    model = AttentionOCR(
        vocabulary=voc, image_width=args.image_width, image_height=args.image_height, max_txt_length=args.max_txt_length, lr=args.learining_rate)

    (trainlength, train_data) = csv_data_source(
        vec, args.data_directory, "train.txt", True)
    (_, validation_data) = csv_data_source(
        vec, args.data_directory, "validation.txt")
    (_, test_data) = csv_data_source(vec, args.data_directory, "test.txt")

    train_gen = tf.data.Dataset.from_generator(
        train_data, output_types=(tf.float32, tf.float32, tf.float32)
    )
    validation_gen = tf.data.Dataset.from_generator(
        validation_data, output_types=(tf.float32, tf.float32, tf.float32)
    )

    if args.pretrained_model:
        model.load(args.pretrained_model)

    validate_every_steps = math.ceil(trainlength / args.batch_size)
    model.fit_generator(
        train_gen,
        validate_every_steps=validate_every_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_gen,
    )

    os.makedirs(os.path.dirname(args.model_name), exist_ok=True)
    model.save(args.model_name)
    os.makedirs('out/Correct', exist_ok=True)
    os.makedirs('out/Wrong', exist_ok=True)
    for image, decoder_input, decoder_output in test_data():
        txt = voc.one_hot_decode(decoder_output, args.max_txt_length)
        (pred, prop) = model.predict([image])
        text = "{}: {:.4f}".format(pred[0], prop[0])
        model.visualise([image], txt)
        print(txt, "prediction: ", text)
