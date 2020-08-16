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

from args_pro import argument_parser

from torchFewShot import models
from torchFewShot.data_manager_pro import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

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

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
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
    model = models.init_model(name=args.arch)
    classifier = models.init_model(name='Proto', scale_cls=args.scale_cls, enable_optim=args.enable_optim, distance=args.distance)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion = CrossEntropyLoss(num_classes=args.n_way, epsilon=args.epsilon)

    param = list(model.parameters()) + \
            list(classifier.parameters())
    optimizer = torch.optim.SGD(param, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False)

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            acc = checkpoint['acc']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- Accuracy: {}".format(args.start_epoch, acc))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    if args.evaluate:
        print('Evaluate only')
        acc = test(model, classifier, testloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_acc_1shot = -np.inf
    best_epoch_1shot = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        train(epoch, model, classifier, criterion, optimizer, trainloader, learning_rate)
        train_time += round(time.time() - start_train_time)
        
        if epoch > (args.stepsize[0]-1) or (epoch + 1) % args.eval_step == 0 or epoch == 0:
            print("==> Test 5-way-1-shot")
            acc_1shot = test(model, classifier, testloader)
            is_best_1shot = acc_1shot > best_acc_1shot

            if is_best_1shot:
                best_acc_1shot = acc_1shot
                best_epoch_1shot = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'acc_1shot': acc_1shot,
                'epoch': epoch,
            }, is_best_1shot, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
            
            print("==> 5-way-1-shot Best accuracy {:.2%}, achieved at epoch {}".format(best_acc_1shot, best_epoch_1shot))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, classifier, criterion, optimizer, trainloader, learning_rate, use_gpu=True):
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, batch in enumerate(trainloader):
        data_time.update(time.time() - end)

        x_support, y_support, x_query, y_query = [x.cuda() for x in batch]
        bs, n1, c, h, w = x_support.size()
        n2 = x_query.size(1)

        f_support = model(x_support.view(-1, c, h, w))
        f_support = f_support.view(bs, n1, -1)

        f_query = model(x_query.view(-1, c, h, w))
        f_query = f_query.view(bs, n2, -1)
        
        outputs = classifier(f_support, y_support, f_query)

        outputs = outputs.view(bs * n2, -1)
        y_query = y_query.view(bs * n2)
        loss = criterion(outputs, y_query)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs.detach().cpu(), 1)
        acc = (torch.sum(preds == y_query.detach().cpu()).float()) / y_query.size(0)
        batch_time.update(time.time() - end)
        losses.update(loss.item(), y_query.size(0))
        accs.update(acc.item(), y_query.size(0))

        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '
          'Acc:{acc.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, 
           loss=losses, acc=accs))


def test(model, classifier, testloader, use_gpu=True):
    accs = AverageMeter()
    model.eval()
    classifier.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            features_train = model(images_train.view(-1, channels, height, width))
            features_train = features_train.view(batch_size, num_train_examples, -1)

            features_test = model(images_test.view(-1, channels, height, width))
            features_test = features_test.view(batch_size, num_test_examples, -1) 

            cls_scores = classifier(features_train, labels_train, features_test)

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
