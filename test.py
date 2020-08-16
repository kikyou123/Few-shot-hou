from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

sys.path.append('./torchFewShot')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from args_cls import argument_parser

from torchFewShot import models
from torchFewShot.data_manager import DataManager
from torchFewShot.ProtoMeasure import few_shot_class

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import set_bn_to_eval, count_num_param, adjust_learning_rate


parser = argument_parser()
args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'), mode='a')
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, scale_cls=args.scale_cls, margin=args.margin, num_classes=args.num_classes)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            print("Loaded checkpoint from '{}'".format(args.resume))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print('Evaluate only')
    acc = test(model, testloader, use_gpu)


def test(model, testloader, use_gpu=True):
    accs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            features_train = model(images_train.view(-1, channels, height, width))
            features_test = model(images_test.view(-1, channels, height, width))
            features_train = features_train.view(batch_size, num_train_examples, -1)
            features_test = features_test.view(batch_size, num_test_examples, -1) 

            cls_scores = few_shot_class(
                features_test=features_test,
                features_train=features_train,
                labels_train=labels_train,
                distance=args.distance) #[batch_size, num_test_examples, nKnovel]

            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

    accuracy = accs.avg
    print('Results --------')
    print('batch_idx: {}, Accuracy: {:.2%}'.format(batch_idx, accuracy))

    return accuracy


if __name__ == '__main__':
    main()
