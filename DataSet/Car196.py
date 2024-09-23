from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image
import pdb
import os
from torchvision import transforms
from collections import defaultdict

from DataSet.CUB200 import MyData, default_loader, Generate_transform_Dict


class Car196:
    def __init__(self, root=None, origin_width=256, width=227, ratio=0.16, transform=None, train_trans='rand-crop',test_trans='center-crop', loader_type='', batch_size=0, debug=False):
        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = '/dataset/image_retrieval/Car196'

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        if loader_type=='':
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], batch_size=batch_size, debug=debug)
        else:
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], loader_type=loader_type, batch_size=batch_size, debug=debug)
        # pdb.set_trace()
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict[test_trans], debug=debug)

        self.query_dict = {
            "query_Rand":MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-rand'], debug=debug),
            'query_0':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], debug=debug),
            'query_90':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-90'], debug=debug),
            'query_180':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-180'], debug=debug),
            'query_270':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-270'], debug=debug)
        }


def testCar196():
    data = Car196()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCar196()
