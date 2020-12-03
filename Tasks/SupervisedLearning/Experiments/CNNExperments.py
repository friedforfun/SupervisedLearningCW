import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
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
        X_train, y_train = gd.get_data(i-1)
        if i == 0:
            class_labels = g_labels[1:]
        else:
            class_labels = binary_labels
        train_s, test_s = run_KFold_experiment(classifier, X_train, y_train, cnn_model_name=classifier_name,
                                               classes_desc=class_title, class_labels=class_labels, stratified=True, custom_name=experiment_name)
        train_scores.append(train_s)
        test_scores.append(test_s)

    return train_scores, test_scores


def run_KFold_experiment(cnn_model, X, y, cnn_model_name='', classes_desc='all-classes', class_labels=g_labels, stratified=False, balance_classes=False, n_splits=10, random_state=0, custom_name=None, **kwargs):
    """Run 1 experiment with K-fold cross validation

    :param cnn_model: A configured instance of a cnn_model inheriting from cnn_model interface
    :type cnn_model: subclass of cnn_model
    :param X: dataset for cross fold validation
    :type X: pandas.DataFrame
    :param y: label data
    :type y: pandas.DataFrane
    :param cnn_model_name: The name of the cnn_model being used to run this experiment, defaults to ''
    :type cnn_model_name: str, optional
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
        Experiment_name = 'SL-KFolds_{}_classifer-{}_stratified-{}'.format(
            classes_desc, cnn_model_name, stratified)
    else:
        Experiment_name = custom_name

    hyperparam_dict = cnn_model.get_params()
    # Run cnn_models using 10-fold cross validation for various learning parameters on the training sets
    if balance_classes:
        X, y = gd.balance_by_class(X, y, random_state=random_state)

    X = X.to_numpy()
    y = y.to_numpy(dtype='int64')

    if stratified:
        # StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        kf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state)
    else:
        # KFold(n_splits=5, shuffle=False, random_state=None)
        kf = KFold(n_splits=n_splits, random_state=random_state)

    train_scores = {}
    test_scores = {}

    for i, (train_indices, test_indices) in enumerate(kf.split(X=X, y=y)):
        with wandb.init(project=Experiment_name, entity='supervisedlearning', reinit=True, config=hyperparam_dict):

            cnn_model.set_wandb_callback(WandbCallback)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            cnn_model.build_classifier(X_train, y_train, (X_test, y_test))
            y_pred = cnn_model.prediction(X_test)
            #y_probs = cnn_model.prediction_proba(X_test)
            #train_scores[i] = cnn_model.run_classifier(X_train, y_train)
            #test_scores[i] = cnn_model.run_classifier(X_test, y_test)
           
            # wandb.log({'Final Accuracy': accuracy_score(
            #     y_test, y_pred), 'Label_class': classes_desc})
            #wandb.sklearn.plot_confusion_matrix(y_test, y_pred, class_labels)

    return train_scores, test_scores
