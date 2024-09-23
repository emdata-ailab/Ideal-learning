from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import math
import pdb
#from . import km_algorithm

    
class MvpLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    
    def __init__(self, alpha = 200.0, beta = 400.0, use_auto_samemargin = False, norm = False, **kwargs):
        super(MvpLoss, self).__init__()
        self.margin = alpha
        self.relative_margin = beta - alpha
        self.use_auto_samemargin  = use_auto_samemargin
        self.kernel = nn.Parameter(torch.Tensor(98,1), requires_grad=True)
        self.norm = norm
        
        if use_auto_samemargin == True:
            self.auto_samemargin = torch.autograd.Variable(torch.Tensor([alpha]).cuda(),requires_grad=True)
        else:
            self.auto_samemargin = alpha    
 
        
    def forward(self, inputs, targets, mode = 'both'):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        self.mode = mode
        simLabel = self.simLabelGeneration(targets)
        GMFlatten, GM = self.calculateGroundMetricContrastive(inputs, simLabel)
        #KM = KM_algorithm(GM.data.cpu().numpy())
        #T = KM.run()
        T = km_algorithm.KM_run(GM.data.cpu().numpy())
        T_flatten = torch.autograd.Variable(torch.from_numpy(T.reshape([-1]))).float().cuda()
        loss = torch.sum(GMFlatten.mul(T_flatten))
        return loss    
        
    def simLabelGeneration(self, targets):
        batch_size = targets.size(0)

        label_expand_batch1 = targets.view(batch_size, 1).repeat(1, batch_size) 
        label_expand_batch2 = targets.view(batch_size, 1).repeat(batch_size, 1) 

        simLabel = torch.eq(label_expand_batch1.view(batch_size*batch_size, -1),
                            label_expand_batch2.view(batch_size*batch_size, -1)).float()
        simLabelMatrix = simLabel.view(batch_size, batch_size)

        return simLabelMatrix

    def calculateGroundMetricContrastive(self, inputs, labelMatrix):
        """
        calculate the ground metric between two batch of features
        """
                # Compute pairwise distance, replace by the official when merged
        batch_size = inputs.size(0)
        if self.norm == False:
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
           # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), self.auto_samemargin + self.relative_margin - dist)                                           
            same_class_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), dist - self.auto_samemargin)
        elif self.norm == True:
            dist = torch.mm(inputs, torch.t(inputs))
            hinge_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), dist - self.auto_samemargin)             
            same_class_groundMetric = torch.max(torch.zeros(batch_size, batch_size).cuda(), self.auto_samemargin + self.relative_margin - dist)            
            
        GM_positivePair = labelMatrix.mul(same_class_groundMetric)
        GM_negativePair = (1 - labelMatrix).mul(hinge_groundMetric)
        
        if self.mode == 'both':
            GM = GM_negativePair + GM_positivePair 
        elif self.mode == 'pos':
            GM = GM_positivePair + (1 - labelMatrix).mul(-10000000.0)
        else:
            GM = GM_negativePair + labelMatrix.mul(-10000000.0)
        GMFlatten = GM.view(-1)
        return GMFlatten, GM
     
class KM_algorithm:
    def __init__(self, groundMetric):
        self.mp = groundMetric
        self.n = groundMetric.shape[0]
        self.link = np.zeros(self.n).astype(np.int16)
        self.lx = np.zeros(self.n)
        self.ly = np.zeros(self.n)
        self.sla = np.zeros(self.n)
        self.visx = np.zeros(self.n).astype(np.bool)
        self.visy = np.zeros(self.n).astype(np.bool)
        
    def DFS(self, x):
        self.visx[x] = True
        for y in range(self.n):
            if self.visy[y]:
                continue
            tmp = self.lx[x] + self.ly[y] - self.mp[x][y]
            if math.fabs(tmp) < 1e-5:
                self.visy[y] = True
                if self.link[y] == -1 or self.DFS(self.link[y]):
                    self.link[y] = x
                    return True
            elif self.sla[y] + 1e-5 > tmp: 
                self.sla[y] = tmp  
        return False
    
    def run(self):
        ################!!!!!!!!!!##############
       # T = np.zeros((self.n, self.n))
       # for index in range(self.n):
       #     T[index][np.argmax(self.mp[index])] = 1.0 / self.n
       # return T
        ################!!!!!!!!!!##############
        
        for index in range(self.n):
            self.link[index] = -1
            self.ly[index] = 0.0
            self.lx[index] = np.max(self.mp[index])
        
        for x in range(self.n):
            self.sla = np.zeros(self.n) + 1e10
            while True:
                self.visx = np.zeros(self.n).astype(np.bool)
                self.visy = np.zeros(self.n).astype(np.bool)
                if self.DFS(x): 
                    break
                d = 1e10
                for i in range(self.n):
                    if self.visy[i] == False:
                        d = min(d, self.sla[i])
                for i in range(self.n):
                    if self.visx[i]:
                        self.lx[i] -= d
                    if self.visy[i]:
                        self.ly[i] += d
                    else:
                        self.sla[i] -= d
        
        res = 0
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            if self.link[i] != -1:
                T[self.link[i]][i] = 1.0 / self.n
        return T

