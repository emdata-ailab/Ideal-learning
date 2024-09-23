import numpy as np
import torch
import time
import random

from collections import OrderedDict
from tqdm import tqdm
import torch

from torch.autograd import Variable

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(self.avg)


def extract_cnn_feature(model, inputs, pool_feature=False):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs)
        # inputs = Variable(inputs).cuda()
        inputs = inputs.cuda()
        if pool_feature is False:
            outputs = model(inputs)
            return outputs
        else:
            # Register forward hook for each module
            outputs = {}


        def func(m, i, o): outputs['pool_feature'] = o.data.view(n, -1)
        hook = model.module._modules.get('features').register_forward_hook(func)
        model(inputs)
        hook.remove()
        # print(outputs['pool_feature'].shape)
        return outputs['pool_feature']


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def extract_features(model, data_loader, metric=None, pool_feature=False):
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


def Recall_at_ks(data='cub', query_feat = None, gallery_feat = None, query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    m = query_feat.shape[0]
    n = gallery_feat.shape[0]
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)


    # Hope to be much faster  yes!!
    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    
    query_ids = torch.from_numpy(gallery_ids) 
    gallery_ids = torch.from_numpy(gallery_ids) 
    for i in range(m):
        x = torch.mm(query_feat[i].reshape(1, -1), gallery_feat.t()).reshape(-1)
        x[i] = 0
        pos_max = torch.max(x[gallery_ids == query_ids[i]])
        neg_num = torch.sum(x + 1e-4 > pos_max) - 1
        neg_nums[i] = neg_num

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]

    return num_valid / float(m)


def evaluate_ideal(epoch, model, query_loader, gallery_loader, data_name, valname=""):
    if data_name in ['shop', 'jd_test']:
        gallery_eq_query = False
        gallery_feature, gallery_labels = extract_features(model, gallery_loader, metric=None, pool_feature=False)
        query_feature, query_labels = extract_features(model, query_loader, metric=None, pool_feature=False)
    
    else:
        gallery_eq_query = True
        features, labels = extract_features(model, query_loader, metric=None, pool_feature=False)
        gallery_feature, gallery_labels = features, labels
        query_feature, query_labels = features, labels

    recall_ks = Recall_at_ks(query_feat = query_feature, gallery_feat = gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=data_name)

    result = '  '.join(['%.4f' % k for k in recall_ks])
    print('Recall: Epoch-%d' % epoch, result, valname)
    return recall_ks
