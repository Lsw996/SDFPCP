import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(self, x, targets):
        self.classes = [0, 1]
        self.x = x
        self.targets = targets
        self.x[np.isnan(self.x)] = 0.

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, targets = self.x[index], self.targets[index]
        return x.astype(np.float32), targets.astype(np.int64)


class CNNDataset(FraudDataset):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.classes = [0, 1]

    def __getitem__(self, index):
        x, targets = self.x[index], self.targets[index]
        x = x.reshape(1, 1034)

        return x.astype(np.float32), targets.astype(np.int64)


def get_dataset(filepath):

    df_raw = pd.read_csv(filepath)
    flags = df_raw.FLAG.copy()

    df_raw.drop(['FLAG'], axis=1, inplace=True)

    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)
    df_raw = df_raw.T.copy()
    df_raw['FLAG'] = flags
    return df_raw


def SGCC_Dataset(filepath):
    df_raw = get_dataset(filepath)
    flags = df_raw['FLAG']
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    df__ = pd.DataFrame(data=df_raw.values, columns=df_raw.columns, index=df_raw.index)
    df__['flags'] = flags

    X = df__.iloc[:, 0:1034].to_numpy()
    target = df__.iloc[:, 1034].to_numpy()

    dataset = CNNDataset(X, target)

    return dataset
