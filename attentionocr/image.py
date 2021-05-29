import cv2
import numpy as np


class ImageUtil:

    def __init__(self, image_height: int, image_width: int):
        '''
            Class wrapper fo loading images and perform preprocess techniques on them before passing the image to the network
                image_height: the max image height
                image_width: the max image width
        '''

        self._image_height = image_height
        self._image_width = image_width

    def load(self, filename: str) -> np.ndarray:
        '''
            read image from the file and conver it to rgb formate
        '''

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess(image)
        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        '''
            Performe preprocess techniques on the image
        '''

        image = self._scale_axis(image)
        image = self._grayscale(image)
        image = self._pad(image)
        image = np.expand_dims(image, axis=2)
        return image

    def _scale_axis(self, image: np.ndarray) -> np.ndarray:
        '''
            Resize the image to have max height and max width
        '''

        height, width, _ = image.shape
        scaling_factor = height / self._image_height
        if height != self._image_height:
            if width / scaling_factor <= self._image_width:
                # scale both axis when the scaled width is smaller than the target width
                image = cv2.resize(image, (int(
                    width / scaling_factor), int(height / scaling_factor)), interpolation=cv2.INTER_AREA,)
            else:
                # otherwise, compress the horizontal axis
                image = cv2.resize(
                    image, (self._image_width, self._image_height), interpolation=cv2.INTER_AREA,)
        elif width > self._image_width:
            # the height matches, but the width is longer
            image = cv2.resize(
                image, (self._image_width, self._image_height), interpolation=cv2.INTER_AREA,)
        return image

    def _grayscale(self, image: np.ndarray) -> np.ndarray:
        '''
            Convert image to gray scale
        '''

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255
        return image

    def _pad(self, image: np.ndarray) -> np.ndarray:
        '''
            Add padding to the image to ensure width and the height are the same
        '''

        result = np.zeros((self._image_height, self._image_width))
        result[:image.shape[0], :image.shape[1]] = image
        return result
