import pandas
from SupervisedLearning.Experiments.GetData import append_result_col, get_data
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
binary_labels = ['yes', 'no']

def run_all_KF_experiments(classifier, classifier_name='', experiment_range=(0, 11), experiment_name='All kf experiments'):
    """Run 10 K-fold for each data label

    :param classifier: A configured instance of a classifier inheriting from Classifier interface.
    :type classifier: a subclass of Classifier.
    :param classifier_name: The name of the classifier. Defaults to ''.
    :type classifier_name: str
    :param experiment_range: The range of experiments to run. Defaults to (0, 11).
    :type experiment_range:  (tuple, optional)

    :return: List of dictionaries with each k-fold scores
    :rtype: list
    """
    train_scores = []
    test_scores = []

    for i in range(experiment_range[0], experiment_range[1]):
        class_title = label_dict[i-1]
        X_train, y_train = get_data(i-1)
        if i == 0:
            class_labels = g_labels[1:]
        else:
            class_labels = binary_labels
        train_s, test_s = run_KFold_experiment(classifier, X_train, y_train, classifier_name=classifier_name, classes_desc=class_title, class_labels=class_labels, stratified=True, custom_name=experiment_name)
        train_scores.append(train_s)
        test_scores.append(test_s)

    return train_scores, test_scores


def run_KFold_experiment(classifier, X, y, classifier_name='', classes_desc='all-classes', class_labels=g_labels, stratified=False, balance_classes=False, n_splits=10, random_state=0, custom_name=None, **kwargs):
    """Run 1 experiment with K-fold cross validation

    :param classifier: A configured instance of a classifier inheriting from Classifier interface
    :type classifier: subclass of Classifier
    :param X: dataset for cross fold validation
    :type X: pandas.DataFrame
    :param y: label data
    :type y: pandas.DataFrane
    :param classifier_name: The name of the classifier being used to run this experiment, defaults to ''
    :type classifier_name: str, optional
    :param classes_desc: A description of the class labels being used, defaults to 'all-classes'
    :type classes_desc: str, optional
    :param class_labels: The labels of each classification in the dataset, defaults to g_labels
    :type class_labels: list(str), optional
    :param stratified: Use stratified k-fold?, defaults to False
    :type stratified: bool, optional
    :param balance_classes: Balance the class distribution, defaults to False
    :type balance_classes: bool, optional
    :param n_splits: number of k-fold splits, defaults to 10
    :type n_splits: int, optional
    :param random_state: Random_state used for ksplits and balancing, defaults to 0
    :type random_state: int, optional
    :return: Tuple of dicts showing scores for each fold
    :rtype: Tuple(dict, dict)
    """
    if custom_name is None:
        Experiment_name = 'SL-KFolds_{}_classifer-{}_stratified-{}'.format(classes_desc, classifier_name, stratified)
    else:
        Experiment_name = custom_name

    hyperparam_dict = classifier.get_params()
    # Run classifiers using 10-fold cross validation for various learning parameters on the training sets
    if balance_classes:
        X, y = gd.balance_by_class(X, y, random_state=random_state, **kwargs)

    X = X.to_numpy()
    y = y.to_numpy(dtype='int64')

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
            wandb.sklearn.plot_confusion_matrix(y_test, y_pred, class_labels)

    return train_scores, test_scores


def run_test_set_experiment(classifier, X_train, X_test, y_train, y_test, classifier_name='', classes_desc='all-classes', class_labels=g_labels):

    Experiment_name = 'SL-Train_Test_{}_classifer-{}'.format(classes_desc, classifier_name)
    hyperparam_dict = classifier.get_params()

    with wandb.init(project=Experiment_name, entity='supervisedlearning', reinit=True, config=hyperparam_dict):
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy(dtype='int64')

        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy(dtype='int64')

        classifier.build_classifier(X_train, y_train)
        y_pred = classifier.prediction(X_test)
        y_probs = classifier.prediction_proba(X_test)
        train_scores = classifier.run_classifier(X_train, y_train)
        test_scores = classifier.run_classifier(X_test, y_test)
        wandb.sklearn.plot_classifier(classifier.get_classifier(),
                                      X_train, X_test, y_train, y_test, y_pred, y_probs, labels=class_labels, model_name=classifier_name)
        #wandb.sklearn.plot_confusion_matrix(y_test, y_pred, class_labels, format='')

    return train_scores, test_scores

def calc_F_measure():
    raise NotImplementedError

def calc_ROC_area():
    raise NotImplementedError

def visualise():
    raise NotImplementedError


def new_test_set(training, testing, num_instances=4000, random_state=0):
    """Create a new test set by taking num_instances from the training set and placing it in the test set.

    Args:
        training (tuple(pandas.DataFrame, pandas.DataFrame)): The training data and corresponding labels
        testing (tuple(pandas.DataFrame, pandas.DataFrame)): The testing data and corresponding labels
        num_instances (int, optional): The number of instances to move from train to test. Defaults to 4000.
        random_state (int, optional): A random state for replicability. Defaults to 0.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame): Training and testing data as: X_train, X_test, y_train, y_test
    """
    train_with_labels = append_result_col(*training)
    test_with_labels = append_result_col(*testing)

    sample = train_with_labels.sample(n=num_instances, random_state=random_state)
    # all indices from sample as list
    sample_indices = list(sample.index)
    train_with_labels = train_with_labels.drop(index=sample_indices)
    
    test_with_labels = test_with_labels.append(sample)

    y_test = train_with_labels[['y']]
    y_train = test_with_labels[['y']]
    X_test = test_with_labels.drop('y', 1)
    X_train = train_with_labels.drop('y', 1)

    return X_test, X_train, y_train, y_test

