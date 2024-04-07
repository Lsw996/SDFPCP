import gc
import sys
import time
import numpy as np
import torch
from configurations import args_parser
from tqdm import trange
import utils
import models
import federated_utils
from torchinfo import summary
from evaluate import evaluate_fn


if __name__ == '__main__':

    args = args_parser()
    boardio, textio, best_val_acc, finally_model = utils.initializations(args)
    textio.cprint(str(args))

    train_data, test_loader = utils.data(args)
    classes = train_data.classes
    labels = np.array(train_data.targets)
    train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    global_model = models.TCN().to(args.device)
    init_FLOPs = summary(global_model, (1, 1, 1034)).total_mult_adds
    global_model.to(args.device)

    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    if args.eval:

        global_model = torch.load(finally_model)
        summary(global_model, (1, 1, 1034), device=args.device)

        test_loss, AUC, ACC = evaluate_fn(global_model, test_loader, train_creterion, args.device)

        textio.cprint(f'AUC:\n {AUC:.4f}')
        textio.cprint(f'ACC:\n {ACC:.4f}')

        gc.collect()
        sys.exit()

    torch.save(global_model, finally_model)

    local_clients = federated_utils.federated_setup(train_data, labels, classes, args)

    total_time = 0
    model_total_size = 0

    FLOPs_list = []
    model_size_list = []
    AUCs = []
    ACCs = []

    for global_epoch in trange(0, args.global_epochs):

        for user_idx in range(args.num_users):
            user_loss = []
            user = local_clients[user_idx]

            federated_utils.distribute_model(user, finally_model, args)

            t = summary(user['model'], (1, 1, 1034), device=args.device)
            FLOPs = t.total_mult_adds
            size = t.total_param_bytes

            R = init_FLOPs / FLOPs

            for local_epoch in range(0, args.local_epochs):
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'], train_creterion,
                                                   args.device)
                if args.lr_scheduler:
                    user['scheduler'].step(train_loss)
                user_loss.append(train_loss)

            if args.prune and args.pT < (global_epoch + 1) and R <= args.R:
                print("\033[1;32m ------------Pruning---------------\033[0m")
                user['model'] = utils.pruning_model(user['model'], args)

            torch.save(user['model'], finally_model)

        valid_loss, AUC, ACC = evaluate_fn(user['model'], val_loader, train_creterion, args.device)

        FLOPs_list.append(FLOPs)
        model_size_list.append(size)
        AUCs.append(AUC)
        ACCs.append(ACC)

        gc.collect()

    print(str(args))
    textio.cprint(f'MACs_list:\n {FLOPs_list}')
    textio.cprint(f'model_size_list:\n {model_size_list}')
    textio.cprint(f'AUCs:\n {AUCs}')
    textio.cprint(f'ACCs:\n {ACCs}')

