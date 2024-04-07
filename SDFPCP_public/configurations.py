import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--eval', default=True)
    parser.add_argument('--iid', default=False)
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_users', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--global_epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--lr_scheduler', default=True)
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=42)
    parser.add_argument('--prune', default=True)
    parser.add_argument('--pl', type=float, default=0.02)
    parser.add_argument('--R', type=float, default=5)
    parser.add_argument('--pT', type=float, default=1)
    parser.add_argument('--cn', type=float, default=16)

    args = parser.parse_args()
    return args

