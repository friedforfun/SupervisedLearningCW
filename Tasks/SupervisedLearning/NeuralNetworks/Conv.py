from sklearn.utils import validation
from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    def __init__(self, num_classes, epochs=50, use_gpu=True, augmented=False):
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
            'augmented': augmented
        }

        self.augmented = augmented
        # if kwargs:
        #     self.params = merge_dict(self.params, kwargs)

        self.epochs = epochs
        self.num_classes = num_classes

        self.wandb_callback = None

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
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=self.num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        self.datagen = ImageDataGenerator(
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=10,
            zoom_range=0.1,  # Randomly zoom image
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images horizontally
            vertical_flip=False)  # Don't randomly flip images vertically

    def augmented_training(self, X_train, y_train, X_test, y_test, batch_size=128):
        """To reduce overfitting we use an image data generator

        Args:
            X_train (numpy.array): training data
            X_test (numpy.array): testing data
            y_train (numpy.array): training labels
            y_test (numpy.array): testing labels
        """
        X_train = self.preprocess_images(X_train)
        y_train = self.y_to_binary(y_train)

        X_test = self.preprocess_images(X_test)
        y_test = self.y_to_binary(y_test)

        self.datagen.fit(X_train)

        if self.wandb_callback is None:
            self.model.fit(self.datagen.flow(X_train, y_train, batch_size=batch_size),  # Default batch_size is 32. We set it here for clarity.
                           epochs=self.epochs,
                        # Run same number of steps we would if we were not using a generator.
                           steps_per_epoch=len(X_train)/batch_size,
                        validation_data=(X_test, y_test))
        else:
            self.model.fit(self.datagen.flow(X_train, y_train, batch_size=batch_size),  # Default batch_size is 32. We set it here for clarity.
                           epochs=self.epochs,
                           # Run same number of steps we would if we were not using a generator.
                           steps_per_epoch=len(X_train)/batch_size,
                           validation_data=(X_test, y_test),
                           callbacks=[self.wandb_callback()])
        

    def conventional_training(self, X_train, y_train, X_test, y_test):

        X_train = self.preprocess_images(X_train)
        X_test = self.preprocess_images(X_test)

        y_test = self.y_to_binary(y_test)
        y_train = self.y_to_binary(y_train)

        if self.wandb_callback is None:
            self.model.fit(X_train, y_train,
                           epochs=self.epochs,
                           verbose=1,
                           validation_data=(X_test, y_test))
        else:
            self.model.fit(X_train, y_train,
                           epochs=self.epochs,
                           verbose=1,
                           validation_data=(X_test, y_test),
                           callbacks=[self.wandb_callback()])


    def preprocess_images(self, image_matrix):
        image_matrix = image_matrix.reshape(-1, 48, 48, 1)
        return image_matrix

    def y_to_binary(self, y):
        return keras.utils.to_categorical(y, self.num_classes)

    def set_wandb_callback(self, callback):
        self.wandb_callback = callback

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

        if self.augmented:
            self.augmented_training(X, y, *validation_data)
        
        else:
            self.conventional_training(X, y, *validation_data)

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
