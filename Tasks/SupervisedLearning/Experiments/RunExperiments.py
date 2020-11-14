from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from . import GetData as gd

def run_experiments(classifier):
    # These experiments for each classifier:
    # all training sets
    # a new set with 4000 instances of the training set moved into the test set
    # a new  set with 9000 instances of the training set moved into the test set
    raise NotImplementedError

def run_experiment(classifier):
    # Step 1:
    # Run classifiers using 10-fold cross validation for various learning parameters on the training sets
    # Step 2:
    # Visualise results
    raise NotImplementedError

def k_folds(classifier):
    X, y = gd.get_data(1)

    scores = []    
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.run_classifier(X_train, y_train)

        scores.append(classifier.score(X_test, y_test))
    return np.mean(scores)

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
        :type X: pandas.dataframe
        :param y: y labels for data
        :type y: pandas.dataframe
        :return: An instance of a confusion matrix from sklearn
        :rtype: confusion_matrix
        """

        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        return confusion_matrix(None, None)


# Example of using inheritence
class NaiveBayse(Classifier):
    def __init__(self, typ='gaussian'):
        if typ == 'gaussian':
            self.classifier = None

    def run_classifier(self, X, y):
        self.classifier.fit(X, y)
        # do classifier stuff in here
        return self.score()

    def score(self):
        y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        return confusion_matrix(y_pred, y_true)