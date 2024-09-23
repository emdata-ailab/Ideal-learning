# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn
from evaluations import Recall_at_ks, pairwise_similarity, extract_features
cudnn.benchmark = True
import numpy as np

def test(epoch, model, query_loader, gallery_loader, args, valname=""):
    if args.data in ['shop', 'jd_test']:
        gallery_eq_query = False
        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=False)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=False)
    
    else:
        gallery_eq_query = True
        features, labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=False)
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels

    recall_ks = Recall_at_ks(query_feat = query_feature, gallery_feat = gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

    result = '  '.join(['%.4f' % k for k in recall_ks])
    print('Recall: Epoch-%d' % epoch, result, valname)
    return recall_ks[0]

def test_multi_queryloader(epoch, model, query_dict, args, valname="", head_type=""):
    recall_dict, features_list = {}, []
    # keys = ['query_0', 'query_90', 'query_180', 'query_270']
    keys = ['query_0', 'query_90', 'query_180', 'query_270']
    for key in keys:
        query_loader = query_dict[key]
        gallery_eq_query = True
        features, labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=False)
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels

        recall_ks = Recall_at_ks(query_feat = query_feature, gallery_feat = gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

        result = '  '.join(['%.4f' % k for k in recall_ks])
        print('Recall: Epoch-%d' % epoch, result, valname, key)
        recall_dict[key] = recall_ks[0]
        features_list.append(features)
    
    features = torch.cat(features_list, dim=1)
    gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    recall_ks = Recall_at_ks(query_feat = query_feature, gallery_feat = gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
    result = '  '.join(['%.4f' % k for k in recall_ks])
    print('Recall: Epoch-%d' % epoch, result, valname, "catFeature recall")
    recall_dict['catFeature'] = recall_ks[0]

    if head_type != "":
        try:
            _, w = features_list[0].shape
            features = torch.cat([features_list[i][:, int(i*w/4):int((i+1)*w/4)] for i in range(4)], dim=1)
            gallery_feature, gallery_labels = query_feature, query_labels = features, labels
            recall_ks = Recall_at_ks(query_feat = query_feature, gallery_feat = gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
            result = '  '.join(['%.4f' % k for k in recall_ks])
            print('Recall: Epoch-%d' % epoch, result, valname, "self")
            recall_dict['Recall'] = recall_ks[0]
        except:
            pass
    return recall_dict
