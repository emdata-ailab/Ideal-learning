from __future__ import print_function, absolute_import

from .NCA import NCALoss
from .Contrastive import ContrastiveLoss
from .Binomial import BinomialLoss
from .LiftedStructure import LiftedStructureLoss
from .Weight import WeightLoss
from .HardMining import HardMiningLoss
from .Softmax import SoftmaxLoss
from .Self import SelfLoss
from .SelfBatch import SelfBatchLoss
from .LimitHinge import LimitHingeLoss
from .Mvp import MvpLoss
from .EccvLoss import EccvLoss
from .Triplet import TripletLoss
from .optLoss import OPTLoss
from .npair_loss import NpairLoss

__factory = {
    'NCA': NCALoss,
    'Contrastive': ContrastiveLoss,
    'Binomial': BinomialLoss,
    'LiftedStructure': LiftedStructureLoss,
    'Weight': WeightLoss,
    'HardMining': HardMiningLoss,
    'Softmax': SoftmaxLoss,
    'Self': SelfLoss,
    'SelfBatch':SelfBatchLoss, 
    'LimitHingeLoss': LimitHingeLoss,
    'EccvLoss': EccvLoss, 
    'Triplet': TripletLoss, 
    'Mvp': MvpLoss,
    'opt': OPTLoss,
    'npair': NpairLoss, 
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
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)


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
        raise KeyError("Unknown loss:", name)
    # print(name,__factory.items())
    return __factory[name]( *args, **kwargs)
