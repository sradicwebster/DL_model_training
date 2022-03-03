# DL model training

Supervised learning framework for training PyTorch models using hydra configuration files and
wandb for tracking and performing hyperparameter tuning. Currently supports training feedfoward neural
network models for classification and regression tasks on torchvision and sklearn datasets.

To train a MLP classifer with default config on the MNIST dataset:

```bash
python main.py +dataset=mnist
```

The default hyperparameters can be changed by passing in additional arguments:

```bash
python main.py +dataset=mnist model.hidden_layer_sizes=[256,256] optimiser=adam optimiser.lr=0.002
```

To train a MLP regressor with default config on the California housing dataset:

```bash
python main.py +dataset=california_housing model.loss_fn=MSELoss model.softmax_output=False
```

To perform hyperparameter tuning (Bayesian optimisation of the optimiser learning rate and weight
decay as defined in the sweep config file) via wandb sweeps:

```bash
python sweep.py --config mnist_lr_wd --count 20
```

See wandb workspace for [here](https://wandb.ai/sradicwebster/hydra-example/sweeps/epfsbgls?workspace=user-sradicwebster).

Work inspired by [Adrish Dey](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw).