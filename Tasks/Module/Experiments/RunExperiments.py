from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
from . import GetData as gd

def run_(classifier):
    X = gd.get_data(1)

    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        pass
    classifier.run_classifier(X, y)

class Classifier(ABC):
    @abstractmethod
    def run_classifier(self, X, y):
        pass


# Example of using inheritence
class NaiveBayse(Classifier):
    def __init__(self, typ='gaussian'):
        if typ == 'gaussian':
            self.classifier = None

    def run_classifier(self, X, y):
        self.classifier.fit(X, y)

        return 'scores'
