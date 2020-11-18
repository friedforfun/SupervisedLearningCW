from sklearn.model_selection import KFold, StratifiedKFold
import wandb
from . import GetData as gd

label_dict = {
    -1: 'All Classes',
    0: 'speed limit 20',
    1: 'speed limit 30',
    2: 'speed limit 50',
    3: 'speed limit 60',
    4: 'speed limit 70',
    5: 'left turn',
    6: 'right turn',
    7: 'beware pedestrian crossing',
    8: 'beware children',
    9: 'beware cycle route ahead'
}

g_labels = [label_dict.get(i) for i in range(-1, 10)]

def run_all_experiments(classifier, project_name='Supervised Learning'):
    # These experiments for each classifier:
    # all training sets
    # a new set with 4000 instances of the training set moved into the test set
    # a new  set with 9000 instances of the training set moved into the test set
    raise NotImplementedError


def run_KFold_experiment(classifier, X, y, classifier_name='', classes_desc='all-classes', class_labels=g_labels, stratified=False, n_splits=10, random_state=0, **kwargs):
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
    Experiment_name = 'SL-KFolds_{}_classifer-{}_stratified-{}'.format(classes_desc, classifier_name, stratified)
    hyperparam_dict = classifier.get_params()
    # Run classifiers using 10-fold cross validation for various learning parameters on the training sets
    X = X.to_numpy()
    y = y.to_numpy()

    if stratified:
        # StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        kf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, **kwargs)
    else:
        # KFold(n_splits=5, shuffle=False, random_state=None)
        kf = KFold(n_splits=n_splits, random_state=random_state,  **kwargs)

    train_scores = {}
    test_scores = {}

    for i, (train_indices, test_indices) in enumerate(kf.split(X=X, y=y)):
        with wandb.init(project=Experiment_name, entity='supervisedlearning', reinit=True, config=hyperparam_dict):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            classifier.build_classifier(X_train, y_train)
            y_pred = classifier.prediction(X_test)
            y_probs = classifier.prediction_proba(X_test)
            train_scores[i] = classifier.run_classifier(X_train, y_train)
            test_scores[i] = classifier.run_classifier(X_test, y_test)
            wandb.sklearn.plot_classifier(classifier.get_classifier(), 
                X_train, X_test, y_train, y_test, y_pred, y_probs, labels=class_labels, model_name=classifier_name)
            
    return train_scores, test_scores

def calc_F_measure():
    raise NotImplementedError

def calc_ROC_area():
    raise NotImplementedError

def visualise():
    raise NotImplementedError

def new_test_set(training, testing, num_instances=4000):
    # move 4000 random from training to testing
    rows_to_move = training.sample(n=num_instances)
    
    return testing.append(rows_to_move, ignore_index=True)

