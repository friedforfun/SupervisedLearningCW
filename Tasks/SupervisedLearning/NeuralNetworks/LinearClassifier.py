from ..Experiments.abstract import Classifier
from sklearn.linear_model import LogisticRegression

class LinearClassifier(Classifier):
    def __init__(self, kind='lr'):

        self.lr = LogisticRegression()


    def run_classifier(self, X, y):

        self.lr.fit(X, y)

        y_pred = self.lr.predict(x_test)

        return self.lr.score(X, y)


    def score(self, X, y):
        pass

