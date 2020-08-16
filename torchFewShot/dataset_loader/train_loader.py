from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io
import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_train(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
        Xtrain: [n_way*n_shot, c, h, w].
        Ytrain: [n_way*n_shot].
        Xtest:  [n_way*query_shot, c, h, w].
        Ytest:  [n_way*query_shot].
    """

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 n_way=5, # number of novel categories.
                 n_shot=1, # number of training examples per novel category.
                 query_shot=6, # number of test examples per novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 load=True,
                 **kwargs
                 ):
        
        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.n_way = n_way
        self.transform = transform
        self.epoch_size = epoch_size

        self.n_shot = n_shot
        self.query_shot = query_shot
        self.load = load

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            support: a list of length 'n_way * n_shot' with 2-element tuples. (sample_index, label)
            query: a list of length 'n_way * query_shot' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.n_way)

        query = []
        support = []
        for Knovel_idx in range(len(Knovel)):
            ids = (self.query_shot + self.n_shot)
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids) 

            imgs_query = img_ids[:self.query_shot]
            imgs_support = img_ids[self.query_shot:]

            query += [(img_id, Knovel_idx) for img_id in imgs_query]
            support += [(img_id, Knovel_idx) for img_id in imgs_support]
        assert(len(query) == self.n_way * self.query_shot)
        assert(len(support) == self.n_way * self.n_shot)
        random.shuffle(support)
        random.shuffle(query)

        return support, query

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [n_shot, c, h, w]
            labels: a tensor [n_shot]
        """

        images = []
        labels = []
        for (img_idx, label) in examples:
            img = self.dataset[img_idx][0]
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        return images, labels

    def __getitem__(self, index):
        support, query = self._sample_episode()
        Xt, Yt = self._creatExamplesTensorData(support)
        Xe, Ye = self._creatExamplesTensorData(query)
        return Xt, Yt, Xe, Ye

