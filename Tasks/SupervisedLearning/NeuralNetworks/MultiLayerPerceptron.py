from ..Experiments.abstract import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

class MultiLayerPerceptron(Classifier):
    """A multilayer perceptron with an interface used for running experiments
    """
    def __init__(self, **kwargs):
        self.mlp = MLPClassifier(**kwargs)

    def get_classifier(self):
        """Get the mlp classifier object

        :return: The configured MLP object
        :rtype: sklearn.neural_network.MLPClassifier
        """
        return self.mlp
    
    def build_classifier(self, X, y):
        """Fit the classifier to the provided data

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        """
        self.mlp.fit(X, y)

    def run_classifier(self, X, y):
        """Make a prediction and return the confusion matrix for a given labelled dataset

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """
        pred = self.mlp.predict(X)

        return confusion_matrix(y, pred)

