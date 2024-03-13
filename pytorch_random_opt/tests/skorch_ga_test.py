import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from skorch import NeuralNetClassifier
from pytorch_random_opt.models import ANNClassifier
from pytorch_random_opt.optimizers import RandomHillClimbing, GeneticAlgorithm, SimulatedAnnealing
from skorch.callbacks import EarlyStopping


if __name__ == '__main__':
    num_of_gpus = torch.cuda.device_count();
    if num_of_gpus > 0:
        use_device = 'cuda'
    else:
        use_device = 'cpu'
    my_random_seed = 995103165
    iris = datasets.load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df['target'] = iris.target

    # Map targets to target names
    target_names = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }

    df['target_names'] = df['target'].map(target_names)
    y_label = df['target_names']
    y_data = df['target']
    X_data = df.drop(['target_names', 'target'], axis=1)
    scaler = RobustScaler()
    X_data_scaled = scaler.fit_transform(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data, stratify=y_data, shuffle=True,
                                                        random_state=my_random_seed)
    dataset = {
        'train': {
            'X': pd.DataFrame(X_train, columns=X_data.columns),
            'y': pd.Series(y_train)
        },
        'test': {
            'X': pd.DataFrame(X_test, columns=X_data.columns),
            'y': pd.Series(y_test)
        }
    }
    n_features = X_train.shape[1]
    n_outputs = y_train.unique().shape[0]

    # showing where these go to make it clear how max_iters and max_attempts map to the stopping criteria here
    max_iters = 200
    max_attempts = 10

    # configuring early stopping callback to function like mlrose stopping criteria
    early_stopping = EarlyStopping(
        monitor='valid_loss',  # Metric to monitor for improvement
        patience=max_attempts,  # Number of epochs to wait before stopping
        threshold=0,  # Minimum change in the monitored metric to qualify as improvement
        threshold_mode='abs',  # Interpretation of the threshold (relative change)
        lower_is_better=True  # Whether a lower metric value is better
    )

    sknet = NeuralNetClassifier(
        ANNClassifier,
        module__n_inputs=n_features,
        module__n_outputs=n_outputs,
        module__activation_func=nn.Tanh,
        module__num_hidden_layers=1,
        module__size_hidden_layers=512,
        max_epochs=max_iters,
        # lr=0.0001,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        device=use_device,
        compile=False,
        optimizer=GeneticAlgorithm,
        optimizer__pop_size=1000
    )

    sknet.fit(X_train, y_train)
