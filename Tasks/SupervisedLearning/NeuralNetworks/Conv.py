from sklearn.utils import validation
from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


def merge_dict(own: dict, other: dict) -> dict:
    """Merge other dict with self

    Args:
        own (dict): the dict being merged into
        other (dict): the dict being read from
    """
    for element in other:
        if own.get(element, None) is None:
            own[element] = other[element]
        else:
            raise ValueError('Conflicting kwargs')
    return own

class ConvolutionalNeuralNetwork(Classifier):
    def __init__(self, num_classes, epochs=50, use_gpu=True, **kwargs):
        if use_gpu:
            self.gpus = tf.config.experimental.list_physical_devices('GPU')
            if self.gpus:
                try:
                    for gpu in self.gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print("Err: ", e)

        self.params = {
            'num_classes': num_classes,
            'epochs': epochs,
        }

        if kwargs:
            self.params = merge_dict(self.params, kwargs)

        self.epochs = epochs
        self.num_classes = num_classes

        self.model = Sequential()
        self.model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(48, 48, 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        self.model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        self.model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=self.num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


    def preprocess_images(self, image_matrix):
        image_matrix = image_matrix.reshape(-1, 48, 48, 1)
        return image_matrix

    def y_to_binary(self, y):
        return keras.utils.to_categorical(y, self.num_classes)


    def run_classifier(self, X, y):
        """Abstract class to use for running experiments

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """
        X = self.preprocess_images(X)

        true_label = self.y_to_binary(y)
        pred_label = self.model.predict(X)
        return confusion_matrix(true_label, pred_label)


    def build_classifier(self, X, y, validation_data=(None, None)):
        """Abstract method to use to fit the classifier to X and y

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        """

        X_train = self.preprocess_images(X)

        if validation_data[0] is not None:
            X_test = self.preprocess_images(validation_data[0])
        else:
            raise ValueError('Validation data must be provided for a CNN')

        y_train = self.y_to_binary(y)
        y_test = self.y_to_binary(validation_data[1])
        
        self.model.fit(X_train, y_train,
                  epochs=self.epochs,
                  verbose=1,
                       validation_data=(X_test, y_test))


    def get_classifier(self):
        """A method to return the classifier object
        """
        return self.model


    def prediction(self, X):
        """Make a prediction given X

        :param X: Data to make prediction on
        :type X: numpy.array
        """
        images = self.preprocess_images(X)
        return self.model.predict(images)


    def prediction_proba(self, X):
        """Find the probabilities for each prediciton

        :param X: Data to make prediction on
        :type X: numpy.array
        """
        images = self.preprocess_images(X)
        return self.model.predict_proba(images)


    def get_params(self):
        """Returns a dictionary of all the classifier hyperparameters
        """
        return self.params
