from .base import BaseDataset
import os
from .utils import Generate_transform_Dict


class Car196:
    def __init__(self, root=None, transform=None, train_trans='rand-crop',test_trans='center-crop', loader_type='', batch_size=0):
        if transform is None:
            transform_Dict = Generate_transform_Dict()
        if root is None:
            root = '/dataset/image_retrieval/Car196'

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        if loader_type=='':
            self.train = BaseDataset(root, label_txt=train_txt, transform=transform_Dict[train_trans], batch_size=batch_size)
        else:
            self.train = BaseDataset(root, label_txt=train_txt, transform=transform_Dict[train_trans], loader_type=loader_type, batch_size=batch_size)

        self.gallery = BaseDataset(root, label_txt=test_txt, transform=transform_Dict[test_trans])

        self.query_dict = \
        {
            "query_Rand":BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-rand']),
            'query_0':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop']),
            'query_90':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-90']),
            'query_180':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-180']),
            'query_270':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-270'])
        }
    

    def nb_classes(self):
        # assert set(self.ys) == set(self.classes)
        return self.train.nb_classes()


def testCar196():
    data = Car196()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCar196()

