"""
Default parameters.
"""


class Config(object):
    """
    Default config (see __main__.py or README for documentation).
    """

    image_width = 320
    image_height = 32
    max_txt_length = 42
    learining_rate = 0  # 0 meen to use default adam optimizer
    epochs = 10
    batch_size = 64
    data_path = 'Data'
    model_name = 'Models/OCR.h5'
    tflite_model_name = 'Models/OCR.tflite'
    LOG_PATH = 'aocr.log'
