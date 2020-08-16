import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).cuda().scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


def get_prototypes(features_train, labels_train):
    """
    Obatain the prototypes of the task.
    Args:
        features_train: [batch_size, num_train_examples, num_channels]
        labels_train: [batch_size, num_train_examples, K] (one-hot)

    Return:
        prototypes: [batch_size, K, num_channels]
    """
    labels_train_transposed = labels_train.transpose(1,2)
    prototypes = torch.bmm(labels_train_transposed, features_train)
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes))
    return prototypes


def apply_cosine(features, prototypes):
    """
    Args: 
        prototypes: [batch_size, K, num_channels]
        features: [batch_size, num_test_examples, num_channels]
    Return:
        cls_scores: [batch_size, num_train_examples, K]
    """
    features = F.normalize( 
        features, p=2, dim=features.dim()-1, eps=1e-12) #[b, n, d]
    prototypes = F.normalize(
        prototypes, p=2, dim=prototypes.dim()-1, eps=1e-12) #[b, k, d]
    cls_scores = torch.bmm(features, prototypes.transpose(1,2))
    return cls_scores


def apply_euclidean(A, B, average=True):
    """
    Args: 
        A: [batch_size, num_test_examples, num_channels]
        B: [batch_size, K, num_channels]
    Return:
        cls_scores: [batch_size, num_test_examples, K]
    """
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    AB = torch.bmm(A, B.transpose(1,2))
    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1) #[batch_size, num_test_examples, 1]
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb) #[batch_size, 1, K]

    dist = (AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB)
    if average:
        dist = dist / nC

    cls_scores = -1 * dist

    return cls_scores
    


def few_shot_class(features_test, features_train, labels_train, distance='euclidean'):
    """
    Classify the test set.
    Args:
        features_test: [batch_size, num_test_examples, num_channels]
        features_train: [batch_size, num_train_examples, num_channels]
        labels_train: [batch_size, num_train_examples] 
    
    Return:
        scores_cls: [batch_size, num_test_examples, K]

    """
    labels_train = one_hot(labels_train) #[batch_size, num_train_examples, K]
    prototypes = get_prototypes(features_train, labels_train) #[batch_size, K, num_channels]
    assert distance in ['cosine', 'euclidean']
    if distance == 'cosine':
        cls_scores = apply_cosine(features_test, prototypes)
    else:
        cls_scores = apply_euclidean(features_test, prototypes)
    return cls_scores


class ProtoNetHead(nn.Module):
    def __init__(self, scale_cls, distance, enable_optim, **kwargs):
        super(ProtoNetHead, self).__init__()
        print('scale_cls:{}, distance:{}, enable_optim: {}'.format(scale_cls, distance, enable_optim))
        self.head = few_shot_class
        self.distance = distance
        if enable_optim:
            self.scale = nn.Parameter(torch.FloatTensor([scale_cls]))
        else:
            self.scale = self.scale_cls

    def forward(self, features_train, labels_train, features_test):
        return self.scale * self.head(features_test, features_train, labels_train, self.distance)
