from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

class RandomForest(Classifier):
    def __init__(self, typ = 'rf'):
        self.rf = RandomForestClassifier(verbose=3)

    def build_classifier(self,X,y):
        self.rf.fit(X,y)

    def run_classifier(self, X, y):
        self.rf.predict(X)

    def build_conf(self, X, y):
        y_pred = self.rf.predict(X)
        return confusion_matrix(y, y_pred)
    
