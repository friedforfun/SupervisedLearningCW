from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from . import GetData as gd



def run_all_experiments(classifier):
    # These experiments for each classifier:
    # all training sets
    # a new set with 4000 instances of the training set moved into the test set
    # a new  set with 9000 instances of the training set moved into the test set
    raise NotImplementedError


def run_experiment(classifier, X, y, stratified=False, n_splits=10, random_state=0,**kwargs):
    """Run 1 experiment with K-fold cross validation

    :param classifier: A configured instance of a classifier inheriting from Classifier interface
    :type classifier: subclass of Classifier
    :param X: dataset for cross fold validation
    :type X: pandas.DataFrame
    :param y: label data
    :type y: pandas.DataFrane
    :param stratified: Use stratified k-fold?, defaults to False
    :type stratified: bool, optional
    :param n_splits: number of k-fold splits, defaults to 10
    :type n_splits: int, optional
    :return: Tuple of dicts showing scores for each fold
    :rtype: Tuple(dict, dict)
    """
    # Run classifiers using 10-fold cross validation for various learning parameters on the training sets
    X = X.to_numpy()
    y = y.to_numpy()

    if stratified:
        # StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        kf = StratifiedKFold(n_splits=n_splits, **kwargs)
    else:
        # KFold(n_splits=5, shuffle=False, random_state=None)
        kf = KFold(n_splits=n_splits, **kwargs)

    train_scores = {}
    test_scores = {}
    for i, (train_indices, test_indices) in enumerate(kf.n_splits(X)):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        classifier.build_classifier(X_train, y_train)
        train_scores[i] = classifier.run_classifier(X_train, y_train)
        test_scores[i] = classifier.run_classifier(X_test, y_test)
    
    return train_scores, test_scores

# Visulisation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py

def calc_F_measure():
    raise NotImplementedError

def calc_ROC_area():
    raise NotImplementedError

def visualise():
    raise NotImplementedError

def new_test_set(num_instances=4000):
    raise NotImplementedError


class Classifier(ABC):
    @abstractmethod
    def run_classifier(self, X, y):
        """Abstract class to use for running experiments

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        #! Make sure the true label is the first arg on confusion matrix and the second is pred
        true_label = None
        pred_label = None
        return confusion_matrix(true_label, pred_label)

    @abstractmethod
    def build_classifier(self, X, y):
        """Abstract method to use to fit the classifier to X and y

        :param X: X data
        :type X: numpy.array
        :param y: y labels for data
        :type y: numpy.array
        """
        pass


# Example of using inheritence
class NaiveBayse(Classifier):
    def __init__(self, typ='gaussian'):
        if typ == 'gaussian':
            self.classifier = None

    def build_classifier(self, X, y):
        self.classifier.fit(X, y)
    

    def run_classifier(self, X, y):
        self.classifier.pred()
        return self.build_conf()

    def build_conf(self):
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        return confusion_matrix(y_pred, y_true)
