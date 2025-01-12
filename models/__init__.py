from .BN_Inception import BN_Inception
from .resnet_office_hist import resnet50

__factory = {
    'BN-Inception': BN_Inception,
    'Resnet':resnet50,
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
