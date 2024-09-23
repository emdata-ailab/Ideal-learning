from .cars import Car196
from .cub import CUB_200_2011
from .SOP import Stanford_Online_Products
import os, pdb

__factory = {
    'car': Car196,
    'cub': CUB_200_2011,
    'SOP': Stanford_Online_Products,
}

def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))
    
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    # pdb.set_trace()
    return __factory[name](root=root, *args, **kwargs)


