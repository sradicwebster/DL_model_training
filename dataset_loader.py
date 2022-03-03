import os
import hydra
import numpy as np
import torch

from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class SklearnDataset(Dataset):
    def __init__(self, data, targets, classes, mean=None, std=None):
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)
        self.classes = classes
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item]
        y = self.targets[item]
        if self.mean is not None:
            X = (X - self.mean) / self.std
        return X, y


def dataset_loader(cfg, seed):
    base_path = hydra.utils.get_original_cwd()

    if cfg.location.split("_")[0] == "pytorch":
        dataset_fn = getattr(datasets, cfg.name)
        train_data = dataset_fn(root=os.path.join(base_path, "data"), train=True, download=True,
                                transform=ToTensor())
        test_data = dataset_fn(root=os.path.join(base_path, "data"), train=False, download=True,
                               transform=ToTensor())

    elif cfg.location.split("_")[0] == "sklearn":
        dataset_fn = getattr(sklearn_datasets, "_".join((cfg.location.split("_")[1],
                                                         cfg.name)))
        data = dataset_fn(data_home=os.path.join(base_path, "data"))
        if cfg.task == "classification":
            classes = data["target_names"]
        else:
            classes = None
        mean = np.mean(data["data"].astype(np.float32), 0)
        std = np.std(data["data"].astype(np.float32), 0)
        X_train, X_test, y_train, y_test = train_test_split(data["data"].astype(np.float32),
                                                            data["target"].astype(np.float32),
                                                            test_size=cfg["test_frac"],
                                                            random_state=seed)
        train_data = SklearnDataset(X_train, y_train, classes, mean, std)
        test_data = SklearnDataset(X_test, y_test, classes, mean, std)

    else:
        print("dataset location must be \"pytorch\" or \"sklearn_\"")

    return train_data, test_data
