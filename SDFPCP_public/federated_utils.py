import torch
import torch.optim as optim
import copy
import math
import utils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset


def federated_setup(train_data, labels, classes, args):

    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}

    if args.iid:
        for user_idx in range(args.num_users):
            user = {'data': torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data,
                                        indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
                batch_size=args.train_batch_size, shuffle=True)}

            local_models[user_idx] = user
    else:

        labels = labels[train_data.indices]
        client_idcs = utils.dirichlet_split_noniid(labels, args)

        n_classes = 10
        n_clients = args.num_users

        plt.figure(figsize=(12, 8))
        label_distribution = [[] for _ in range(n_classes)]
        for c_id, idc in enumerate(client_idcs):
            for idx in idc:
                label_distribution[labels[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, n_clients + 1.5, 1),
                 label=classes, rwidth=0.5)
        plt.xticks(np.arange(n_clients), ["Client %d" %
                                          c_id for c_id in range(n_clients)])
        plt.xlabel("Client ID")
        plt.ylabel("Number of samples")
        plt.legend()
        plt.title("Display Label Distribution on Different Clients")
        plt.savefig(f'fig/K={args.num_users}_Non-iid_alpha={args.alpha}.png')

        for user_idx in range(args.num_users):
            user = {'data': torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data, client_idcs[user_idx]),
                batch_size=args.train_batch_size, shuffle=True)}

            local_models[user_idx] = user

    return local_models


def distribute_model(user, finally_model, args):

    user['model'] = torch.load(finally_model)
    user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                            momentum=args.momentum) if args.optimizer == 'sgd' \
        else optim.Adam(user['model'].parameters(), lr=args.lr)
    if args.lr_scheduler:
        user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
            user['opt'], patience=10, factor=0.1, verbose=True)
