import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ...NeuralNetworks.MultiLayerPerceptron import MultiLayerPerceptron
from ..RunExperiments import run_KFold_experiment, g_labels
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
    model = MultiLayerPerceptron(max_iter=config.max_iter, hidden_layer_sizes=config.hidden_layer_sizes)

    X, y = gd.get_data()
    X, y = gd.balance_by_class(X, y, random_state=0)

    X = X.to_numpy()
    y = y.to_numpy(dtype='int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

    model.build_classifier(X_train, y_train)
    y_pred = model.prediction(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics inside your training loop
    metrics = {'accuracy': accuracy, 'loss': model.loss_}
    wandb.log(metrics)




if __name__ == '__main__':
   main()
