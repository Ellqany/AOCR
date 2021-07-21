# Attention OCR

A clear and maintainable implementation of Attention OCR in Tensorflow 2.0.

This sequence to sequence OCR model aims to provide a clear and maintainable implementation of attention based OCR.

This repository depends upon the following:

- Tensorflow 2.4.1
- Python 3.7+

## Acknowledgements

This project is basiclly converting Ed Medvedev implementation to tensorflow 2. You can find the original model in the [emedvedev/attention-ocr](https://github.com/emedvedev/attention-ocr) repository. His project is based on a model by [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03). You can find the original model in the [da03/Attention-OCR](https://github.com/da03/Attention-OCR) repository.

## The model

Authors: [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03).

The model first runs a sliding CNN on the image (images are resized to height 32 while preserving aspect ratio). Then an LSTM is stacked on top of the CNN. Finally, an attention model is used as a decoder for producing the final outputs.

![OCR example](http://cs.cmu.edu/~yuntiand/OCR-2.jpg)

## Usage

### Create a dataset

To train the model, you need a collection of images and an annotation file with their respective labels.

Annotations are simple text files containing the image paths (either absolute or relative to your working dir) and their corresponding labels:

```
datasets/images/hello.jpg hello
datasets/images/world.jpg world
```

### Train

```
aocr train --data_directory=Data
```

A new model will be created, and the training will start. Note that it takes quite a long time to reach convergence, since we are training the CNN and attention model simultaneously.

The model save a snapshot after each epoch and it will be saved in the dir `snapshots/`.
After the training the model will be saved in (the default output dir is `Models/ocr.h5`).

**Important:** there is a lot of available training options. See the CLI help or the `parameters` section of this README.

### Test and visualize

```
aocr test --data_directory=Data --model_name=models/ocr.h5
```

Additionally, you can visualize the attention results during testing saved to `out/` by default:

```
aocr test --visualize --data_directory=Data --model_name=models/ocr.h5
```

Example output images in `results/correct`:

Image 0 (j/j):

![example image 0](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_0.jpg)

Image 1 (u/u):

![example image 1](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_1.jpg)

Image 2 (n/n):

![example image 2](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_2.jpg)

Image 3 (g/g):

![example image 3](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_3.jpg)

Image 4 (l/l):

![example image 4](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_4.jpg)

Image 5 (e/e):

![example image 5](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_5.jpg)

### Convert To TfLite

```
aocr tflite --tf_model=models/ocr2.h5 --model_name=ocr.tflite
```

this convert tensorflow model to tflite tf_model is the location to tensorflow model and model_name is the location to store tflite model in.

## Parameters

### Global

* `LOG_PATH`: Path for the log file.

### Testing

* `visualize`: Output the attention maps on the original image.
* `learining_rate`: Initial learning rate, note the we use AdaDelta, so the initial value does not matter much.
* `data_directory`: The location of test data(`test.txt`) in txt format.
* `image_width`: Maximum width for the input images. WARNING: images with the width higher than maximum will be discarded.
* `image_height`: Maximum height for the input images.
* `max_txt_length`: Maximum length of the predicted word/phrase.

### Training

* `epochs`: The number of whole data passes.
* `batch_size`: Batch size.
* `learining_rate`: Initial learning rate, note the we use AdaDelta, so the initial value does not matter much.
* `data_directory`: The location of training data(`train.txt`) and validation data(`validation.txt`) the validation data not required.
* `image_width`: Maximum width for the input images. WARNING: images with the width higher than maximum will be discarded.
* `image_height`: Maximum height for the input images.
* `max_txt_length`: Maximum length of the predicted word/phrase.

### Convert To TfLite

* `learining_rate`: Initial learning rate, note the we use AdaDelta, so the initial value does not matter much.
* `image_width`: Maximum width for the input images. WARNING: images with the width higher than maximum will be discarded.
* `image_height`: Maximum height for the input images.
* `max_txt_length`: Maximum length of the predicted word/phrase.

## References

- [tensorflow 1 attention-ocr](https://github.com/emedvedev/attention-ocr)

- [What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938.pdf)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## To do

- Add qunization when export to tflite
- Make installer
