from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import os, pdb
import sys
# import transforms
from DataSet import transforms 
from collections import defaultdict


def default_loader(path):
    return Image.open(path).convert('RGB')

def Generate_transform_Dict(origin_width=256, width=227, ratio=0.16):
    
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    transform_dict['center-crop-90'] = transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.RotateAnticlockwise(90),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    transform_dict['center-crop-180'] = transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.RotateAnticlockwise(180),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    transform_dict['center-crop-270'] = transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.RotateAnticlockwise(270),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['center-crop-rand'] = transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.RotateAnticlockwise(rotate=[0,90,180,270]),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((width)),
                    transforms.ToTensor(),
                    normalize,
                ])

    transform_dict['rand-crop-90'] = transforms.Compose([
                transforms.CovertBGR(),
                transforms.RandomHorizontalFlip(),
                transforms.RotateAnticlockwise(90),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.ToTensor(),
                normalize,
               ]) 

    transform_dict['rand-crop-180'] = transforms.Compose([
                transforms.CovertBGR(),
                transforms.RandomHorizontalFlip(),
                transforms.RotateAnticlockwise(180),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['rand-crop-270'] = transforms.Compose([
                transforms.CovertBGR(),
                transforms.RandomHorizontalFlip(),
                transforms.RotateAnticlockwise(270),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['rand-crop-rand'] = transforms.Compose([
                transforms.CovertBGR(),
                transforms.RandomHorizontalFlip(),
                transforms.RotateAnticlockwise(rotate=[0,90,180,270]),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.ToTensor(),
                normalize,
               ]) 
    
    return transform_dict
 

class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader, loader_type='', batch_size=0, debug=False):
        # loader_type: batch_single_angle, batch_multi_angle
        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "/dataset/image_retrieval/Car196"
        self.root = root
        self.debug = debug
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
        # print(f'>>>>classes:{classes}')

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
        # fn = os.path.join(self.root, fn)
        fn = self.root + fn
        # print(f'====>fn:{fn}')
        # pdb.set_trace()
        img = self.loader(fn)
        if self.transform is not None and self.loader_type == "":
            img = self.transform(img)
            return img, label
        elif 'repeat' not in self.loader_type:
            # print(self.last_transform_index, self.sample_count, self.rotate_index, self.cell_instance, self.rotate_index // self.cell_instance)
            if self.last_transform_index !=  self.rotate_index // self.cell_instance:
                self.last_transform_index =  self.rotate_index // self.cell_instance
                self.transform_index = self.get_transform_index()
            img = self.transform[self.transform_index](img)
            self.rotate_index += 1
            return img, label, self.transform_index, index
        else:
            # pdb.set_trace()
            img, transform_inds = self.get_repeat_tensor(img)
            # pdb.set_trace()
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

    def __len__(self):
        if self.debug:
            return self.batch_size
        else:
            return len(self.images)


class CUB_200_2011:
    def __init__(self, width=227, origin_width=256, ratio=0.16, root=None, transform=None, train_trans='rand-crop',test_trans='center-crop', loader_type='', batch_size=0, debug=False):
        print('width: \t {}'.format(width))
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = "/dataset/image_retrieval/CUB_200_2011/"

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        if loader_type=='':
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], batch_size=batch_size, debug=debug)
        else:
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], loader_type=loader_type, batch_size=batch_size, debug=debug)
        self.gallery = MyData(root, label_txt=test_txt,             transform=transform_Dict[test_trans], debug=debug)
        self.query_dict = {
            "query_Rand":MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-rand'], debug=debug),
            'query_0':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], debug=debug),
            'query_90':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-90'], debug=debug),
            'query_180':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-180'], debug=debug),
            'query_270':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-270'], debug=debug)
        }

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def testCUB_200_2011():
    print(CUB_200_2011.__name__)
    data = CUB_200_2011()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCUB_200_2011()