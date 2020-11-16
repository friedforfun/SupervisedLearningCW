from ..Experiments.RunExperiments import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LinearClassifier(Classifier):
    def __init__(self, kind='lr', **kwargs):
        self.lr = LogisticRegression(**kwargs)

    def build_classifier(self, X, y):
        self.lr.fit(X, y)

    def run_classifier(self, X, y):
        pred = self.lr.predict(X)
        return confusion_matrix(y, pred)

    def score(self, x, y):
        # X_test, y_test
        return self.lr.score(x, y)
