from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from . import GetData as gd

def run_experiments(classifier):
    X, y = gd.get_data(1)

    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        pass
    classifier.run_classifier(X, y)

class Classifier(ABC):
    @abstractmethod
    def run_classifier(self, X, y):
        """Abstract class to use for running experiments

        :param X: X data
        :type X: pandas.dataframe
        :param y: y labels for data
        :type y: pandas.dataframe
        :return: 4-tuple of confusion matrix
        :rtype: (int, int, int, int)
        """
        true_positive = None
        false_positive = None
        false_negative = None
        true_negative = None
        return (true_positive, false_positive, false_negative, true_negative)


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
        return (0,0,0,0)