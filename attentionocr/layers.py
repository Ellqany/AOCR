import math
import tensorflow as tf

from typing import Tuple
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dropout, MaxPool2D, Conv2D, LSTM, BatchNormalization, Dense


class Encoder:

    layers = [
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

        Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        Conv2D(512, (2, 2), padding='valid', activation='relu',
               kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(rate=0.5)
    ]

    def __init__(self, units: int):
        '''
            The feature extraction model
                units: the number of units to be existed in lstm layer mus be even
        '''

        assert units % 2 == 0  # units must be even, because the encoder is bidirectional
        self.cnn = Sequential(self.layers)
        self.lstm = Bidirectional(LSTM(units // 2, return_sequences=True))

    def __call__(self, encoder_input: list) -> tf.Tensor:
        out = self.cnn(encoder_input)
        if(out.shape[1] == 1):
            out = tf.squeeze(out, axis=1)
        else:
            # Reshape the out put
            shape = tf.shape(out)
            out = tf.reshape(
                out, (shape[0], 1, out.shape[1] * out.shape[2], out.shape[3]))
            out = tf.squeeze(out, axis=1)

        self.cnn_shape = out.shape
        lstm = self.lstm(out)
        return lstm

    @staticmethod
    def get_width(width):
        '''
            get the width of the netowrk
        '''
        for layer in Encoder.layers:
            if type(layer) is MaxPool2D:
                assert layer.strides == layer.pool_size
                width = math.ceil(width / layer.pool_size[1])
            elif type(layer) is Conv2D and layer.padding == 'valid':
                assert layer.strides == (1, 1)
                width = width - math.ceil(layer.kernel_size[1] / 2.0)
        return width


class Attention:

    def __init__(self, units: int):
        '''
            the Attention layer
        '''
        self.query_projection = Dense(units)

    def __call__(self, decoder_input, encoder_output) -> Tuple[tf.Tensor, tf.Tensor]:
        query = self.query_projection(decoder_input)
        key = encoder_output
        value = encoder_output
        logits = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        attention_weights = tf.nn.softmax(logits)
        context_vectors = tf.matmul(attention_weights, value)
        return (context_vectors, attention_weights)


class Decoder:

    def __init__(self, units):
        '''
            the Decoder layer
        '''
        self.lstm = LSTM(units, return_sequences=True, return_state=True)

    def __call__(self, decoder_input, initial_state) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.lstm(decoder_input, initial_state)


class DecoderOutput:
    def __init__(self, vocab_size):
        self.hidden = Dense(256, activation='relu')
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, decoder_output) -> tf.Tensor:
        net = self.hidden(decoder_output)
        return self.dense(net)
