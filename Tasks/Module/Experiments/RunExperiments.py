from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from . import GetData as gd

def run_experiments(classifier):
    X, y = gd.get_data(1)

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):

        classifier.run_classifier(X, y)

class Classifier(ABC):
    @abstractmethod
    def run_classifier(self, X, y):
        """Abstract class to use for running experiments

        :param X: X data
        :type X: pandas.dataframe
        :param y: y labels for data
        :type y: pandas.dataframe
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        return confusion_matrix(None, None)


# Example of using inheritence
class NaiveBayse(Classifier):
    def __init__(self, typ='gaussian'):
        if typ == 'gaussian':
            self.classifier = None

    def run_classifier(self, X, y):
        self.classifier.fit(X, y)
        # do classifier stuff in here
        return self.score()

    def score(self):
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        return confusion_matrix(y_pred, y_true)