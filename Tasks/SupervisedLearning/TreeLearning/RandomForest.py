from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import graphviz

class RandomForest(Classifier):
    """A Random Forest with an interface used for running experiments
    """
    def __init__(self, **kwargs):
        self.hyper_params = kwargs
        self.rf = RandomForestClassifier(**kwargs)
    
    def get_params(self):
        return self.hyper_params

    def run_classifier(self, X, y):
            """Make a prediction and return the confusion matrix for given labelled dataset

            :param X: X data
            :type X: numpy.array
            :param y: y labels for data
            :type y: numpy.array
            :return: An instance of a confusion matrix from sklearn
            :rtype: confusion_matrix
            """
            y_pred = self.rf.predict(X)
            return confusion_matrix(y, y_pred)

    def build_classifier(self,X,y):
        """Fit the classifier to the provided data
        
        :param X: X data
        :type X: numpy.array
        :param y: labels for data
        :type y: numpy array
        """
        self.rf.fit(X,y)

    def get_classifier(self):
        """A method to return the classifier object 
        """
        return self.rf

    def prediction(self,X):
        """Make a prediction given X

        :param X: Data to make prediction on
        :type X: numpy.array
        """
        y_pred = self.rf.predict(X)
        return y_pred

    def prediction_proba(self,X):
        """Find the probablitities for each prediction

        :param X: Data to make prediction on
        :type X: numpy.array
        """
        return self.rf.predict_proba(X)    
