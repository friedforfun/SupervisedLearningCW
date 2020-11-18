from ..Experiments.abstract import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class LinearClassifier(Classifier):
    """A Linear Classifier with an interface used for running experiments
    """
    def __init__(self, kind='lr', **kwargs):
        self.hyper_params = kwargs
        self.lr = LogisticRegression(**kwargs)

    def get_params(self):
        return self.hyper_params

    def build_classifier(self, X, y):
        """Fit the classifier to the provided data

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        """
        self.lr.fit(X, y)

    def run_classifier(self, X, y):
        """Make a prediction and return the confusion matrix for a given labelled dataset

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """
        pred = self.lr.predict(X)
        return confusion_matrix(y, pred)

    def get_classifier(self):
        """Get the lr classifier object

        :return: The configured LR object
        :rtype: sklearn.linear_model.LogisticRegression
        """
        return self.lr

    def prediction(self, x):
        """Infer labels given X

        :param X: Data to make prediction on
        :type X: numpy.array
        :return: An array of labels
        :rtype: numpy.array
        """
        return self.lr.predict(x)

    def prediction_proba(self, X):
        return self.lr.predict_proba(X)

    def score(self, x, y):
        # X_test, y_test
        return self.lr.score(x, y)
