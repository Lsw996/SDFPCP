import os
from statistics import mean
import torch
from tensorboardX import SummaryWriter
import numpy as np
import torch_pruning as tp
import dataset


def data(args):

    SGCC_dataset = dataset.SGCC_Dataset('./data/dataset.csv')
    train_len = round(len(SGCC_dataset) * 0.6)
    train_data, test_data = torch.utils.data.random_split(SGCC_dataset, [train_len, len(SGCC_dataset) - train_len],
                                                          generator=torch.Generator().manual_seed(0))
    train_data.classes = 2
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=True)

    return train_data.dataset, test_loader


def data_split(data, amount, args):

    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    return train_data, val_loader


def train_one_epoch(train_loader, model,
                    optimizer, creterion,
                    device):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device).float()

        label = label.to(device)
        model = model.to(device)
        output = model(data)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return mean(losses)


def initializations(args):

    torch.backends.cudnn.deterministic = True

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    finally_model = 'model/finally_model.pth'

    return boardio, textio, best_val_acc, finally_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def pruning_model(model, args):

    imp = tp.importance.MagnitudeImportance(p=1)

    example_inputs = torch.randn(1, 1, 1034)
    example_inputs = example_inputs.to(args.device)

    ignored_layers = []
    for m in model.modules():
        if (isinstance(m, torch.nn.Conv1d) and m.out_channels <= args.cn) or (
                isinstance(m, torch.nn.Linear) and m.out_features <= args.cn):
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=args.pl,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    return model


def dirichlet_split_noniid(train_labels, args):

    n_clients = args.num_users
    alpha = args.alpha
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

