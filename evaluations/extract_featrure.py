from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
from tqdm import tqdm
import torch
from utils import to_numpy
import numpy as np

from utils.meters import AverageMeter
from evaluations.cnn import extract_cnn_feature


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def extract_features(model, data_loader, print_freq=1, metric=None, pool_feature=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    feature_gpu = torch.FloatTensor().cuda()

    trans_inter = 1e4
    labels = list()
    end = time.time()
    # data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, data in enumerate(data_loader):
        if len(data) == 2:
            imgs, pids = data
        else:
            imgs, pids, _, _ = data
            if imgs.size(1) == 12:
                imgs=torch.cat(imgs.chunk(4,1),dim=0)
        outputs = extract_cnn_feature(model, imgs, pool_feature=pool_feature)
        feature_gpu = torch.cat((feature_gpu, outputs.data), 0)
        labels.extend(pids)
        count = feature_gpu.size(0)
        if count % trans_inter == 0 or i == len(data_loader)-1:
            data_time.update(time.time() - end)
            end = time.time()

            batch_time.update(time.time() - end)
            print('Extract Features: [{}/{}]'
                  .format(i + 1, len(data_loader)))

            end = time.time()
        del outputs
    
    return feature_gpu, labels


def pairwise_distance(features, metric=None):
    n = features.size(0)
    # normalize feature before test
    x = normalize(features)
    # print(4*'\n', x.size())
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    # print(dist.size())
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) 
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(x, y=None):
    if y is None:
        y = x 
    # normalization
    y = normalize(y)
    x = normalize(x)
    # similarity
    similarity = torch.mm(x, y.t())
    return similarity
