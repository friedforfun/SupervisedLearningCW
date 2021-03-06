import wandb
import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Tasks.SupervisedLearning.Experiments import GetData as gd

#! REPLACE THIS WITH YOUR CLASSIFIER
from Tasks.SupervisedLearning.TreeLearning.RandomForest import RandomForest


#! Fill in all areas encapsulated in `----``
#
# Then as described in step 3 here:  https://docs.wandb.com/sweeps/quickstart
# 
# Define your .yaml file
# Run
# $ wandb sweep <path to .yaml>
# 
# Copy the command output by the above command and run it.


def parse_args():
    """Specify all args to be passed into the script here

    :return: Argument parser with config
    :rtype: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    #!----------------------- STEP 1: ADD AGRUMENTS HERE ---------------------------------
    # Follow this syntax, the first arg cannot be '-h' the second arg should be the same as its key in hyperparam_defaults
    # dest is the attribute used to pass into the classifier
    # type should be defined
    parser.add_argument('-est', '--n_estimators', nargs=1,
                        help='Number of trees in forest', dest='n_estimators', type=int)
    parser.add_argument('-verb', '--verbose', nargs=1,
                        help='Verbosity', dest='verbose', type=int)
    parser.add_argument('-crit', '--criterion', nargs=1,
                        help='criterion - quality of split', dest='criterion', type=str)
    parser.add_argument('-mf', '--max_features', nargs=1,
                        help='considered no of features before split', dest='max_features', type=str)                   
    parser.add_argument('-mss', '--min_samples_split', nargs=1,
                        help='min no of samples to split internal node', dest='min_samples_split', type=int)
    parser.add_argument('-msl', '--min_samples_leaf', nargs=1,
                        help='min no of samples required at leaf node', dest='min_samples_leaf', type=int)
    parser.add_argument('-mwfl', '--min_weight_fraction_leaf', nargs=1,
                        help='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.', dest='min_weight_fraction_leaf', type=float)
    #! -----------------------------------------------------------------------------------
    return parser.parse_args()


#! ---------------------------STEP 2:  SET ABS PATH TO DATA -------------------------------
# This should be the absolute path to the data folder
# Dont forget to extract the rar file in the repository
DATA_FOLDER = '/home/sam/Projects/SupervisedLearningCW/Tasks/Data/'

#! ----------------------------------------------------------------------------------------



#! ---------------------------STEP 3:  DEFINE HYPERPARAMETERS HERE-------------------------
# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparam_defaults = {
    'n_estimators': 100,
    'criterion': 'gini',
    'max_features' : 'auto',
    'min_samples_split' : 2,
    'min_samples_leaf' : 1
}
#! ----------------------------------------------------------------------------------------

# Pass defaults to wandb.init
wandb.init(config=hyperparam_defaults, entity='supervisedlearning', project='SL-sweeps')

config = wandb.config


def main():
    run(parse_args())


def run(args):
    try:
        #! ---------------STEP 4: APPLY ALL ARGS TO CLASSIFIER --------------------
        # Type of these args is a list, value at index 0 is the arg
        n_estimators = args.n_estimators[0]
        criterion = args.criterion[0]
        max_features = args.max_features[0]
        min_samples_split = args.min_samples_split[0]
        min_samples_leaf = args.min_samples_leaf[0]

        # pass arg into classifer
        #! Assign your classifer to the var `model`
        model = RandomForest(oob_score=True,
           n_estimators=n_estimators, verbose=0, criterion=criterion, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        #! ----------------------------------------------------------------------
        X, y = gd.get_data(
            root_path=DATA_FOLDER)
        X, y = gd.balance_by_class(X, y, random_state=0)

        X = X.to_numpy()
        y = y.to_numpy(dtype='int64').flatten()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.33)

        model.build_classifier(X_train, y_train)
        y_pred = model.prediction(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics inside your training loop
        metrics = {'accuracy': accuracy, 'oob_score_': model.get_classifier().oob_score_}
        wandb.log(metrics)

    except TypeError as e:
        display_exception(e)
        wandb.alert(title="Sweep exception raised",
                    text="Check the types of the parameters.\n {}".format(e))
        sys.exit(1)
    except AttributeError as e:
        display_exception(e)
        wandb.alert(title="Sweep exception raised",
                    text="Check the attributes being called in script (line 66?) \n {}".format(e))
        sys.exit(1)
    except Exception as e:
        display_exception(e)
        wandb.alert(title="Sweep exception raised",
                    text="An exception was raised during this sweep. \n {}".format(e))
        sys.exit(1)


def display_exception(e):
    print('Exception encountered: ', e)
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(e).__name__, e.args)
    print(message)


if __name__ == '__main__':
   sys.exit(main() or 0)
