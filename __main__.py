import os
import sys
import argparse
import tensorflow as tf

from defaults import Config
from attentionocr import (Vectorizer, AttentionOCR, Vocabulary)
from utility import train_attention_ocr, test_attention_ocr

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.get_logger().setLevel('ERROR')


def process_args(args, defaults):
    parser = argparse.ArgumentParser()
    parser.prog = 'aocr'
    subparsers = parser.add_subparsers()

    # Parse Model integers
    parser_model = argparse.ArgumentParser(add_help=False)
    parser_model.add_argument(
        '--image_width', type=int, default=defaults.image_width, required=False,
        help=('max image width (default: %s)' % (defaults.image_width)))

    parser_model.add_argument(
        '--image_height', type=int, default=defaults.image_height, required=False,
        help=('max image height (default: %s)' % (defaults.image_height)))

    parser_model.add_argument(
        '--max_txt_length', type=int, default=defaults.max_txt_length, required=False,
        help=('max length of predicted strings (default: %s)' % (defaults.max_txt_length)))

    parser_model.add_argument(
        '--learining_rate', type=float, default=0, required=False,
        help=('initial learning rate (default: %s)' % (defaults.learining_rate)))

    # Training
    parser_train = subparsers.add_parser(
        'train', parents=[parser_model], help='Train the model and save checkpoints.')
    parser_train.set_defaults(phase='train')
    parser_train.add_argument('--epochs', type=int, default=10, required=False,
                              help=('number of training epochs (default: %s)' % (defaults.epochs)))

    parser_train.add_argument('--batch_size', type=int, default=64, required=False,
                              help=('batch size (default: %s)' % (defaults.batch_size)))

    parser_train.add_argument('--data_directory', type=str, default=defaults.data_path, required=True,
                              help=('training dataset in the txt format, default=%s' % (defaults.data_path)))

    parser_train.add_argument('--pretrained_model', type=str, default=None, required=False,
                              help=('the location of the pretrainned model'))

    parser_train.add_argument('--model_name', type=str, default=defaults.model_name, required=True,
                              help=('the complete model path to store the model in after trainning, default=%s' % (defaults.model_name)))

    # Testing
    parser_test = subparsers.add_parser(
        'test', parents=[parser_model], help='Test the saved model.')
    parser_test.set_defaults(phase='test')

    parser_test.add_argument('--data_directory', type=str, default=defaults.data_path, required=True,
                             help=('testing dataset in the txt format, default=%s' % (defaults.data_path)))
    parser_test.add_argument('--model_name', type=str, default=defaults.model_name, required=True,
                             help=('the complete model path, default=%s' % (defaults.model_name)))
    parser_test.add_argument('--visualize', dest='visualize',
                             action='store_true', help=('visualize attentions'))

    parser_tflite = subparsers.add_parser(
        'tflite', parents=[parser_model], help='export the model in tflite formate.')
    parser_tflite.set_defaults(phase='tflite')
    parser_tflite.add_argument("--tf_model", type=str, default=defaults.model_name, required=True,
                               help=('the complete model path, default=%s' % (defaults.model_name)))
    parser_tflite.add_argument('--model_name', type=str, default=defaults.tflite_model_name, required=True,
                               help=('the complete model path to store the model in after coverting to tflites, default=%s' % (defaults.tflite_model_name)))

    return parser.parse_args(args)


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)

    voc = Vocabulary()
    vec = Vectorizer(
        vocabulary=voc, image_width=parameters.image_width, image_height=parameters.image_height, max_txt_length=parameters.max_txt_length)
    model = AttentionOCR(
        vocabulary=voc, image_width=parameters.image_width, image_height=parameters.image_height, max_txt_length=parameters.max_txt_length, lr=parameters.learining_rate)

    if parameters.phase == 'train':
        if parameters.pretrained_model:
            model.load(parameters.pretrained_model)

        train_attention_ocr(model=model, vec=vec, model_name=parameters.model_name, data_directory=parameters.data_directory,
                            batch_size=parameters.batch_size, epochs=parameters.epochs)
    elif parameters.phase == 'test':
        if os.path.isfile(parameters.model_name):
            model.load_models(parameters.model_name)
        test_attention_ocr(model=model, voc=voc, vec=vec, visualization=parameters.visualize,
                           data_directory=parameters.data_directory)
    elif parameters.phase == 'tflite':
        if os.path.isfile(parameters.tf_model):
            model.load_models(parameters.tf_model)

        model.export2tflite(parameters.model_name)
    else:
        print('Phase: ' + parameters.phase)
        raise NotImplementedError


if __name__ == "__main__":
    main()
