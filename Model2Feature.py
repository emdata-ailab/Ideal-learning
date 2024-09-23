# coding=utf-8
from __future__ import absolute_import, print_function
import pdb
import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, nThreads=16, batch_size=100, pool_feature=False, testAllAngle=False, test_trans='center-crop', **kargs):
    dataset_name = data
    model = models.create(net, dim=dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)
    resume = checkpoint
    model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(data, width=width, root=root, test_trans=test_trans)
    
    if dataset_name in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=nThreads)

        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        pdb.set_trace()
        if not testAllAngle:
            data_loader = torch.utils.data.DataLoader(
                data.gallery, batch_size=batch_size,
                shuffle=False, drop_last=False, pin_memory=True,
                num_workers=nThreads)
            features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
            gallery_feature, gallery_labels = query_feature, query_labels = features, labels
        else:
            keys = ['query_0', 'query_90', 'query_180', 'query_270']
            features_list = []
            for key in keys:
                data_loader = torch.utils.data.DataLoader(data.query_dict[key], batch_size=batch_size,shuffle=False, drop_last=False,pin_memory=True,num_workers=nThreads)
                features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
                features_list.append(features)
            features = torch.cat(features_list, dim=1) / 2
            # features = (features_list[0]+features_list[1]+features_list[2]+features_list[3])/2
            gallery_feature, gallery_labels = query_feature, query_labels = features, labels

    return gallery_feature, gallery_labels, query_feature, query_labels
