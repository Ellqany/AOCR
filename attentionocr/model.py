import os
import datetime
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import Image
from tensorflow.keras import Input
from attentionocr import metrics, Vocabulary, Encoder, Attention, Decoder, DecoderOutput


class AttentionOCR:

    def __init__(self, vocabulary: Vocabulary, image_height=32, image_width=320, max_txt_length: int = 42, lr=0, units: int = 256):
        self._vocabulary = vocabulary
        self._max_txt_length = max_txt_length
        self._image_height = image_height
        self._image_width = image_width
        self._units = units

        if(lr > 0):
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam()

        self.stats = {}
        log_dir = os.path.join(
            "logs", datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        self.tensorboard_writer = tf.summary.create_file_writer(logdir=log_dir)

        # Build the model.
        self._encoder_input = Input(
            shape=(self._image_height, self._image_width, 1), name="encoder_input")
        self._decoder_input = Input(
            shape=(None, len(self._vocabulary)), name="decoder_input")

        self._encoder = Encoder(self._units)
        self._attention = Attention(self._units)
        self._decoder = Decoder(self._units)
        self._output = DecoderOutput(len(self._vocabulary))

        self._training_model = self.build_training_model()
        self._inference_model = self.build_inference_model()
        self._visualisation_model = self.build_inference_model(
            include_attention=True)

    def build_training_model(self) -> tf.keras.Model:
        encoder_output = self._encoder(self._encoder_input)

        batch_size = tf.shape(self._decoder_input)[0]
        eye = tf.eye(self._max_txt_length,
                     batch_shape=tf.expand_dims(batch_size, 0))
        attention_input = tf.concat([self._decoder_input, eye], axis=2)

        context_vectors, _ = self._attention(attention_input, encoder_output)
        x = tf.concat([self._decoder_input, context_vectors], axis=2)
        decoder_output, _, _ = self._decoder(x, initial_state=None)
        logits = self._output(decoder_output)
        return tf.keras.Model([self._encoder_input, self._decoder_input], [logits])

    def build_inference_model(self, include_attention: bool = False) -> tf.keras.Model:
        predictions = []
        attentions = []
        prediction = self._decoder_input
        encoder_output = self._encoder(self._encoder_input)
        batch_size = tf.shape(self._decoder_input)[0]
        eye = tf.eye(self._max_txt_length,
                     batch_shape=tf.expand_dims(batch_size, 0))
        initial_state = None
        for i in range(self._max_txt_length):
            position = tf.expand_dims(eye[:, :, i], axis=1)
            attention_input = tf.concat([prediction, position], axis=2)
            context_vectors, attention = self._attention(
                attention_input, encoder_output)
            attentions.append(attention)
            x = tf.concat([prediction, context_vectors], axis=2)
            decoder_output, hidden_state, cell_state = self._decoder(
                x, initial_state=initial_state)
            initial_state = [hidden_state, cell_state]
            prediction = self._output(decoder_output)
            predictions.append(prediction)
        predictions = tf.concat(predictions, axis=1)
        attentions = tf.concat(attentions, axis=1)
        output = [predictions, attentions] if include_attention else predictions
        return tf.keras.Model([self._encoder_input, self._decoder_input], output)

    def fit_generator(self, generator, validate_every_steps: int, epochs: int = 1, batch_size: int = 64, validation_data=None) -> None:
        for epoch in range(1, epochs + 1):
            batches = generator.batch(batch_size)
            pbar = tqdm(batches)
            pbar.set_description("Epoch %03d / %03d " % (epoch, epochs))
            for batch in pbar:
                loss = self._training_step(*batch)
                self.stats["training loss"] = "%.4f" % loss
                self.stats["iterations"] = self.optimizer.iterations.numpy()

                if self.optimizer.iterations % validate_every_steps == 0 and validation_data is not None:
                    accuracies = []
                    for validation_batch in validation_data.batch(batch_size):
                        accuracy = self._validation_step(*validation_batch)
                        accuracies.append(accuracy)
                    self.stats["test accuracy"] = np.mean(accuracies)

                pbar.set_postfix(self.stats)

            self.save('snapshots/snapshot-%d.h5' % epoch)

    def _training_step(self, x_image: np.ndarray, x_decoder: np.ndarray, y_true: np.ndarray) -> float:
        if x_decoder.shape[1] == 1:
            raise ValueError(
                "Please provide training data during training (set is_training=True)")
        with tf.GradientTape() as tape:
            y_pred = self._training_model([x_image, x_decoder], training=True)
            loss = self._calculate_loss(y_true, y_pred)
            self._update_tensorboard(train_loss=loss)
        variables = self._training_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()

    def _validation_step(self, x_image: np.ndarray, x_decoder: np.ndarray, y_true: np.ndarray) -> float:
        if x_decoder.shape[1] != 1:
            raise ValueError(
                "Please provide validation data (set is_training=False)")
        # determine the test accuracy using the inference model
        y_pred = self._inference_model([x_image, x_decoder], training=False)
        accuracy = metrics.masked_accuracy(y_true, y_pred)
        # Update the tensorboards
        self._update_tensorboard(accuracy=accuracy)
        return accuracy

    @staticmethod
    def _calculate_loss(y_true, y_pred) -> tf.Tensor:
        loss = metrics.masked_loss(y_true, y_pred)
        return loss

    def _update_tensorboard(self, **kwargs):
        with self.tensorboard_writer.as_default():
            for name in kwargs:
                tf.summary.scalar(
                    name, kwargs[name], step=self.optimizer.iterations)

    def save(self, filepath) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._training_model.save_weights(filepath=filepath)

    def load(self, filepath) -> None:
        self._training_model.load_weights(filepath=filepath)

    def predict(self, images):
        texts = []
        propapilities = []
        for image in images:
            image = np.expand_dims(image, axis=0)

            decoder_input = np.zeros((1, 1, len(self._vocabulary)))
            decoder_input[0, :, :] = self._vocabulary.one_hot_encode(
                '', 1, sos=True, eos=False)

            y_pred = self._inference_model.predict([image, decoder_input])

            if y_pred.shape[-1] > 1:
                val_max = np.argmax(y_pred[0][0])
                propapilities.append(y_pred[0][0][val_max])

        y_pred = np.squeeze(y_pred, axis=0)  # squeeze the batch index out
        texts.append(self._vocabulary.one_hot_decode(
            y_pred, self._max_txt_length))
        return (texts, propapilities)

    def visualise(self, images, corecttext: str):
        for image in images:
            input_image = np.expand_dims(image, axis=0)

            decoder_input = np.zeros((1, 1, len(self._vocabulary)))
            decoder_input[0, :, :] = self._vocabulary.one_hot_encode(
                '', 1, sos=True, eos=False)

            y_pred, attention = self._visualisation_model.predict(
                [input_image, decoder_input])
            y_pred = np.squeeze(y_pred, axis=0)
            text = self._vocabulary.one_hot_decode(
                y_pred, self._max_txt_length)

            step_size = float(image.shape[1]) / attention.shape[-1]
            img_out_frames = []
            for index, char_idx in enumerate(np.argmax(y_pred, axis=-1)):
                if self._vocabulary.is_special_character(char_idx):
                    break
                heatmap = np.zeros(image.shape)
                for location, strength in enumerate(attention[0, index, :]):
                    heatmap[:, int(
                        location * step_size): int((location + 1) * step_size)] = strength * 255.0

                filtered_image = image * 255 * 0.4 + heatmap * 0.6
                filtered_image = (filtered_image).astype(np.float)
                filtered_image = np.squeeze(filtered_image, axis=2)
                filtered_image = Image.fromarray(filtered_image)
                filtered_image = filtered_image.convert('RGB')
                img_out_frames.append(filtered_image)

            if (text == corecttext):
                dirpath = os.path.join('out/Correct/', text + '.gif')
            else:
                dirpath = os.path.join('out/Wrong/', text + '.gif')

            img_out_frames[0].save(dirpath, format='GIF', save_all=True,
                                   loop=0, duration=200, append_images=img_out_frames[1:])
