from .base import *
import os
import os.path as osp
from .utils import Generate_transform_Dict

class Stanford_Online_Products:
    def __init__(self, root=None, transform=None, train_trans='rand-crop',test_trans='center-crop', loader_type='', batch_size=0):
        transform_Dict = Generate_transform_Dict()
        if root is None:
            root = '/dataset/image_retrieval/Stanford_Online_Products/'
        # import pdb; pdb.set_trace()
        train_txt = osp.join(root, 'train.txt')
        test_txt = osp.join(root, 'test.txt')
        if loader_type=='':
            self.train = BaseDataset(root, label_txt=train_txt, transform=transform_Dict[train_trans], batch_size=batch_size)
        else:
            self.train = BaseDataset(root, label_txt=train_txt, transform=transform_Dict[train_trans], loader_type=loader_type, batch_size=batch_size)
        self.gallery = BaseDataset(root, label_txt=test_txt, transform=transform_Dict[test_trans])

        self.query_dict = {
            "query_Rand":BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-rand']),
            'query_0':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop']),
            'query_90':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-90']),
            'query_180':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-180']),
            'query_270':BaseDataset(root, label_txt=test_txt, transform=transform_Dict['center-crop-270'])
        }
    
    
    def nb_classes(self):
        return self.train.nb_classes()
    
def test():
    data = Products()
    print(data.train[1][0][0][0])
    print(len(data.gallery), len(data.train))


if __name__=='__main__':
    test()
