from __future__ import absolute_import

import os
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from DataSet.CUB200 import default_loader, Generate_transform_Dict, MyData


class Products:
    def __init__(self, width=224, origin_width=256, ratio=0.16, root=None, transform=None, train_trans='rand-crop',test_trans='center-crop', loader_type='', batch_size=0, debug=False):
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = '/dataset/image_retrieval/Stanford_Online_Products/'
        # import pdb; pdb.set_trace()
        train_txt = osp.join(root, 'train.txt')
        test_txt = osp.join(root, 'test.txt')
        if loader_type=='':
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], batch_size=batch_size, debug=debug)
        else:
            self.train = MyData(root, label_txt=train_txt, transform=transform_Dict[train_trans], loader_type=loader_type, batch_size=batch_size, debug=debug)
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict[test_trans], debug=debug)

        self.query_dict = {
            "query_Rand":MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-rand'], debug=debug),
            'query_0':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], debug=debug),
            'query_90':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-90'], debug=debug),
            'query_180':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-180'], debug=debug),
            'query_270':MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop-270'], debug=debug)
        }
    
def test():
    data = Products()
    print(data.train[1][0][0][0])
    print(len(data.gallery), len(data.train))


if __name__=='__main__':
    test()




