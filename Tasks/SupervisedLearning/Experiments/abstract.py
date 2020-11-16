from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def run_classifier(self, X, y):
        """Abstract class to use for running experiments

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        #! Make sure the true label is the first arg on confusion matrix and the second is pred
        true_label = None
        pred_label = None
        return confusion_matrix(true_label, pred_label)

    @abstractmethod
    def build_classifier(self, X, y):
        """Abstract method to use to fit the classifier to X and y

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        """
        pass

    @abstractmethod
    def get_classifier(self):
        """A method to return the classifier object
        """
        pass


# Example of using inheritence
class NaiveBayse(Classifier):
    def __init__(self, typ='gaussian'):
        if typ == 'gaussian':
            self.classifier = None

    def build_classifier(self, X, y):
        self.classifier.fit(X, y)

    def run_classifier(self, X, y):
        self.classifier.pred()
        return self.build_conf()

    def build_conf(self):
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        return confusion_matrix(y_pred, y_true)
