import wandb
import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Tasks.SupervisedLearning.Experiments import GetData as gd

#! REPLACE THIS WITH YOUR CLASSIFIER
from Tasks.SupervisedLearning.NeuralNetworks.MultiLayerPerceptron import MultiLayerPerceptron


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
    parser.add_argument('-mi', '--max_iter', nargs=1,
                        help='Max iterations', dest='max_iter', type=int)
    parser.add_argument('-hls', '--hidden_layer_sizes', nargs=1,
                        help='Hidden layer size', dest='hidden_layer_sizes', type=int)

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
    'max_iter': 100,
    'hidden_layer_sizes': 200
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
        max_iter = args.max_iter[0]
        hidden_layer_sizes = args.hidden_layer_sizes[0]

        # pass arg into classifer
        #! Assign your classifer to the var `model`
        model = MultiLayerPerceptron(
            max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
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
        metrics = {'accuracy': accuracy, 'loss': model.mlp.loss_}
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
