from ..Experiments.abstract import Classifier
from sklearn.metrics import confusion_matrix
from sklearn import tree as j48
import graphviz

class J48(Classifier):
    def __init__(self):
        self.j48 = j48.DecisionTreeClassifier()

        def build_classifier(self,X,y):
            self.j48.fit(X,y)

        def run_classifier(self, X, y):
            self.j48.predict(X)
            
        def build_conf(self, X, y):
            y_pred = self.j48.predict(X)
            return confusion_matrix(y, y_pred)
        
        def print_tree():
            dot_data = j48.export_graphviz(self.j48, out_file=None) 
            graph = graphviz.Source(dot_data) 
            graph.render("j48_graph") 
                        
            
