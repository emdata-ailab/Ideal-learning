from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
import torch.utils.data as data
from PIL import Image
import random
import os, pdb
import sys
from .utils import *
from collections import defaultdict


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, label_txt=None,transform=None, loader=default_loader, loader_type='', batch_size=0):
        # loader_type: batch_single_angle, batch_multi_angle
        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "/dataset/image_retrieval/Car196"
        self.root = root
        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = Generate_transform_Dict()['rand-crop']

        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.loader_type = loader_type
        if loader_type == "":
            self.transform = transform
        else:
            self.transform = [Generate_transform_Dict()['rand-crop'],
            Generate_transform_Dict()['rand-crop-90'],
            Generate_transform_Dict()['rand-crop-180'],
            Generate_transform_Dict()['rand-crop-270']]
            if 'multi' in self.loader_type:
                self.cell_instance = int(batch_size/4)
            else:
                self.cell_instance = batch_size
            # pdb.set_trace()
            self.rand = 'rand' in self.loader_type
            if self.rand:
                self.sample_list = [random.randint(0,3) for _ in range(1024)]
                self.sample_count = 0
            self.rotate_index = 0
            self.last_transform_index = self.rotate_index // self.cell_instance
            self.transform_index = self.get_transform_index()
            print("self.transform_index", self.transform_index)
            self.inds = np.arange(4)
        
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = self.root + fn
        img = self.loader(fn)
        if self.transform is not None and self.loader_type == "":
            img = self.transform(img)
            return img, label
        elif 'repeat' not in self.loader_type:
            
            if self.last_transform_index !=  self.rotate_index // self.cell_instance:
                self.last_transform_index =  self.rotate_index // self.cell_instance
                self.transform_index = self.get_transform_index()
            img = self.transform[self.transform_index](img)
            self.rotate_index += 1
            return img, label, self.transform_index, index
        else:

            img, transform_inds = self.get_repeat_tensor(img)

            return img, label, transform_inds, index
    
    def get_repeat_tensor(self, img):
        if self.rand:
            np.random.shuffle(self.inds)
        img_list = [self.transform[ind](img.copy()) for ind in self.inds]
        return torch.cat(img_list, dim=0), torch.from_numpy(self.inds)
    
    def get_transform_index(self):
        if not self.rand:
            return self.last_transform_index % 4
        else:
            indx = self.sample_list[self.sample_count % 1024]
            self.sample_count += 1
            return indx

    def nb_classes(self):
        # assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.images)
