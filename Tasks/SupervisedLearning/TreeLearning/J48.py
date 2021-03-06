from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
from sklearn import tree as j48
import graphviz

class J48(Classifier):
    """ J48 implementation used via an interface for running experiments 

    :param Classifier: [description]
    :type Classifier: [type]
    """
    

    def __init__(self, **kwargs):
        self.hyper_params = kwargs
        self.j48 = j48.DecisionTreeClassifier(**kwargs)

    def build_classifier(self,X,y, validation_data=(None, None)):
        """Fit the classifier to the provided data

        :param X: X data
        :type X: numpy.array
        :param y: labels for data
        :type y: numpy.array
        """
        self.j48.fit(X,y)

    def run_classifier(self, X, y):
        """ Used to run the J48 tree

        :param X: [description]
        :type X: [type]
        :param y: [description]
        :type y: [type]
        :return: [description]
        :rtype: [type]
        """
        
        pred = self.j48.predict(X)
        return confusion_matrix(y, pred)
            
    def prediction(self, X):
        """Make a prediction given X

        :param X:  Data to make prediction on
        :type X: numpy.array
        """
        
        return self.j48.predict(X)
    
    def prediction_proba(self, X):
        """Find the probablitities for each prediction

        :param X: Data to make prediction on
        :type X: numpy.array
        """
        
        return self.j48.predict_proba(X)
        
    def get_params(self):
        """

        :return: [description]
        :rtype: [type]
        """
        
        return self.hyper_params
    
    def get_classifier(self):
        """

        :return: [description]
        :rtype: [type]
        """
        
        return self.j48
                            
    def print_tree():
        """ Print the tree
        """
        
        dot_data = j48.export_graphviz(self.j48, out_file=None) 
        graph = graphviz.Source(dot_data) 
        graph.render("j48_graph") 
            
