import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet_load')
    parser.add_argument('--load', default=True)

    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--height', type=int, default=84)
    parser.add_argument('--width', type=int, default=84)

    # ************************************************************
    # Data Augument
    # ************************************************************
    parser.add_argument('--erasing_p', type=float, default=0)

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-04, type=float)

    parser.add_argument('--max-epoch', default=30, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--stepsize', default=[20], nargs='+', type=int)
    parser.add_argument('--LUT_lr', default=[(20, 0.1), (27, 0.006), (30, 0.0012)])

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--train-batch', default=8, type=int)
    parser.add_argument('--test-batch', default=4, type=int)

    parser.add_argument('--n_way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--n_shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--train_query_shot', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--train_epoch_size', type=int, default=1000 * 8,
                            help='number of episodes per training epoch')

    parser.add_argument('--query_shot', type=int, default=15)
    parser.add_argument('--epoch_size', type=int, default=600)

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--epsilon', default=0)
    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('--arch', type=str, default='ResNet12')
    parser.add_argument('--save-dir', type=str, default='./result/miniImageNet_cls/proto/bs8-consine-s7')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0, 2', type=str)

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--distance', default='cosine', type=str)
    parser.add_argument('--scale_cls', type=int, default=7)
    parser.add_argument('--enable_optim', default=True)
    # ************************************************************

    # Test settings
    # ************************************************************
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--start-eval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--phase', default='test', type=str)

    return parser

