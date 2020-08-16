import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet_myload')
    parser.add_argument('--load', default=True)

    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--height', type=int, default=84)
    parser.add_argument('--width', type=int, default=84)

    # ************************************************************
    # Data Augument
    # ************************************************************
    parser.add_argument('--erasing_p', type=float, default=0.5)

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-04, type=float)

    parser.add_argument('--max-epoch', default=80, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int)
    parser.add_argument('--LUT_lr', default=[(60, 0.1), (70, 0.006), (80, 0.0012)])

    parser.add_argument('--train-batch', default=128, type=int)
    parser.add_argument('--test-batch', default=4, type=int)

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--epsilon', default=0)
    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('--arch', type=str, default='ResNet12')
    parser.add_argument('--cls-arch', type=str, default='AngleLinear')
    parser.add_argument('--save-dir', type=str, default='./result/miniImageNet_cls/norm_erase0.5_bs128/cat64-s16-myload')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='3', type=str)

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=64)
    parser.add_argument('--scale_cls', type=int, default=16)
    # ************************************************************

    # Test settings
    # ************************************************************
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--start-eval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--query_shot', type=int, default=30)
    parser.add_argument('--epoch_size', type=int, default=600)

    parser.add_argument('--distance', default='cosine', type=str)
    parser.add_argument('--phase', default='test', type=str)

    return parser

