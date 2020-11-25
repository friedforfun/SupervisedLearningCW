import wandb
import sys, argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ...NeuralNetworks.MultiLayerPerceptron import MultiLayerPerceptron
from .. import GetData as gd

# Step 3
# https://docs.wandb.com/sweeps/quickstart

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparam_defaults = {
    'max_iter': 100,
    'hidden_layer_sizes': 200
}

# Pass your defaults to wandb.init
wandb.init(config=hyperparam_defaults)

config = wandb.config

def main():
    run(parse_args())

def run(args):
    try:
        max_iter = args.max_iter
        hidden_layer_sizes = args.hidden_layer_sizes

        model = MultiLayerPerceptron(
            max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)

        X, y = gd.get_data()
        X, y = gd.balance_by_class(X, y, random_state=0)

        X = X.to_numpy()
        y = y.to_numpy(dtype='int64')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0, test_size=0.33)

        model.build_classifier(X_train, y_train)
        y_pred = model.prediction(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics inside your training loop
        metrics = {'accuracy': accuracy, 'loss': model.loss_}
        wandb.log(metrics)
    except Exception as e:
        print('Exception encountered: ', e)
        wandb.alert(title="Sweep exception raised",
                    text="An exception was raised during this sweep. \n {}".format(e))
        sys.exit(1)
    

def parse_args():
    """Specify all args to be passed into the script here

    :return: Argument parser with config
    :rtype: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    args = parser.add_argument_group('hyperparams')
    args.add_argument('-i', '--max_iter', nargs=1,help='Max iterations')
    args.add_argument('-h', '--hidden_layer_sizes', nargs=1, help='Hidden layer size')
    return parser.parse_args()


if __name__ == '__main__':
   sys.exit(main() or 0)
