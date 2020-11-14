from ..Experiments.RunExperiments import Classifier
from sklearn.linear_model import LogisticRegression

class LinearClassifier(Classifier):
    def __init__():
        pass

    def run_classifier(self, X, y):
        # lr = LogisticRegression()
        # lr.fit(X, y)

        self.classifier.fit(X, y)

        y_pred = lr.predict(x_test)

        return lr.score(X, y)
        return confusion_matrix(y_test, y_pred)

    def score(self):
        return confusion_matrix(y_test, y_pred)

