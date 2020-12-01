import wandb
import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Tasks.SupervisedLearning.Experiments import GetData as gd

#! REPLACE THIS WITH YOUR CLASSIFIER
from Tasks.SupervisedLearning.TreeLearning.J48 import J48


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
    parser.add_argument('-crit', '--criterion', nargs=1,
                        help='criterion - quality of split', dest='criterion', type=str)
    parser.add_argument('-split', '--splitter', nargs=1,
                        help='splitter - strategy used to choose the split at each node', dest='splitter', type=str)
    parser.add_argument('-md', '--max_depth', nargs=1,
                        help='maximum depth of the tree', dest='max_depth', type=int)
    parser.add_argument('-mss', '--min_samples_split', nargs=1,
                        help='min no of samples to split internal node', dest='min_samples_split', type=int)
    parser.add_argument('-msl', '--min_samples_leaf', nargs=1,
                        help='The minimum number of samples required to be at a leaf node', dest='min_samples_leaf', type=int)
    parser.add_argument('-mwfl', '--min_weight_fraction_leaf', nargs=1,
                        help='minimum weighted fraction of the sum total of weights required to be at a leaf node', dest='min_weight_fraction_leaf', type=float)
    parser.add_argument('-mf', '--max_features', nargs=1,
                        help='considered no of features before split', dest='max_features', type=str)     
    parser.add_argument('-rs', '--random_state', nargs=1,
                        help='Controls both the randomness of the bootstrapping', dest='random_state', type=int)
    parser.add_argument('-mln', '--max_leaf_nodes', nargs=1,
                        help='Grow a tree with max_leaf_nodes in best-first fashion', dest='max_leaf_nodes', type=int)
    parser.add_argument('-mid', '--min_impurity_decrease', nargs=1,
                        help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value', dest='min_impurity_decrease', type=float)
             
    

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
    'criterion': 'gini',
    'splitter': 'best',
    'max_depth': 1,
    'min_samples_split' : 2,
    'min_samples_leaf' : 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features' : 'auto',
    'random_state': 1,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0
    
}
#! ----------------------------------------------------------------------------------------

# Pass defaults to wandb.init
wandb.init(config=hyperparam_defaults)

config = wandb.config


def main():
    run(parse_args())


def run(args):
    try:
        #! ---------------STEP 4: APPLY ALL ARGS TO CLASSIFIER --------------------
        # Type of these args is a list, value at index 0 is the arg
        criterion = args.criterion[0]
        splitter = args.splitter[0]
        max_depth = args.max_depth[0]
        min_samples_split = args.min_samples_split[0]
        min_samples_leaf = args.min_samples_leaf[0]
        min_weight_fraction_leaf = args.min_weight_fraction_leaf[0]
        max_features = args.max_features[0]
        random_state = 1
        max_leaf_nodes = None
        min_impurity_decrease = args.min_impurity_decrease[0]
        
        # pass arg into classifer
        #! Assign your classifer to the var `model`
        model = J48(
            criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease)
        #! ----------------------------------------------------------------------
        X, y = gd.get_data(
            root_path=DATA_FOLDER)
        X, y = gd.balance_by_class(X, y, random_state=0)

        X = X.to_numpy()
        y = y.to_numpy(dtype='int64')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.33)

        model.build_classifier(X_train, y_train)
        y_pred = model.prediction(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics inside your training loop
        metrics = {'accuracy': accuracy}
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
