# encoding: utf-8
"""
@author:  Zhiyuan Chen
@contact: <zhiyuanchen01@gmail.com>
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

class OPTLoss(nn.Module): 
    def __init__(self, args=None,  **kwargs): 
        super(OPTLoss, self).__init__()
        self.m_lb = args.opt_marginlb  # cfg.MODEL.LOSSES.OPT.MARGINLB  # 0.4  不同类的余弦相似度的最小值
        self.m_ub = args.opt_marginub  # cfg.MODEL.LOSSES.OPT.MARGINUB  # 0.6  同类的余弦相似度最大值
        self._scale = args.opt_scale  # cfg.MODEL.LOSSES.OPT.SCALE  # 1
        self.dist_pos = 0.0
        self.dist_neg = 0.0
        # self.last_time = -1
        self.kernel = nn.Parameter(torch.Tensor(98,1), requires_grad=True)
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)  

    def forward(self, embedding, targets, mode = None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        embedding = F.normalize(embedding, dim=1)
        dist_mat = torch.matmul(embedding, embedding.t())
        
        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(targets.view(M, 1).expand(M, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, M).ne(targets.view(M, 1).expand(M, N).t()).float()

        if self.dist_pos == 0.0:
            self.dist_pos = (torch.sum(dist_mat * is_pos) - N) / (torch.sum(is_pos) - N)
        else:
            self.dist_pos = self.dist_pos * 0.99 + 0.01 * (torch.sum(dist_mat * is_pos) - N) / (torch.sum(is_pos) - N)
        if self.dist_neg == 0.0:
            self.dist_neg = torch.sum(dist_mat * is_neg) / torch.sum(is_neg)
        else:
            self.dist_neg = self.dist_neg * 0.99 + 0.01 * (torch.sum(dist_mat * is_neg) / torch.sum(is_neg))

        cost_pos = is_pos.mul( (dist_mat > self.m_ub).float() + dist_mat) + is_neg.mul(30.0)
        cost_pos = (cost_pos - torch.min(cost_pos.detach())) / (torch.max(cost_pos.detach()) - torch.min(cost_pos.detach()))

        cost_neg = is_neg.mul( (dist_mat < self.m_lb).float() + 1.0 - dist_mat) + is_pos.mul(30.0) 
        cost_neg = (cost_neg - torch.min(cost_neg.detach())) / (torch.max(cost_neg.detach()) - torch.min(cost_neg.detach()))
        
        edge_pos = is_pos.mul(F.relu(self.m_ub - dist_mat)) + is_neg.mul(-1e7)   
        edge_neg = is_neg.mul(F.relu(dist_mat - self.m_lb)) + is_pos.mul(-1e7)
      
        opt_pos = sinkhorn_loss(cost_pos, 100.0, N, 100); #print('torch.sum(opt_pos,dim = 1) is', torch.sum(opt_pos,dim = 1))
        opt_neg = sinkhorn_loss(cost_neg, 100.0, M, 100); #print('torch.sum(opt_neg,dim = 1) is', torch.sum(opt_neg,dim = 1))
        
        pos_num = torch.sum(opt_pos.mul(N * is_pos * (dist_mat < self.m_ub).float()) ).item()
        neg_num = torch.sum(opt_neg.mul(N * is_neg * (dist_mat > self.m_lb).float()) ).item()
        loss = torch.sum(opt_pos * edge_pos) + torch.sum(opt_neg * edge_neg)

        # cur_time = int(int(time.time()) / 100)   
        # if self.last_time != cur_time:
        #     self.last_time = cur_time
        #     print('dist_pos:', self.dist_pos.item(), 'dist_neg:', self.dist_neg.item())
        #     print('opt pos sum is', torch.sum(dist_mat.mul(is_pos) * opt_pos).item())
        #     print('torch.sum(opt_pos,dim = 1) is', torch.min(torch.sum(opt_pos,dim = 1)).item(), torch.max(torch.sum(opt_pos,dim = 1)).item())
        #     print('opt neg sum is', torch.sum(dist_mat.mul(is_neg) * opt_neg).item())
        #     print('torch.sum(opt_neg,dim = 1) is', torch.min(torch.sum(opt_neg,dim = 1)).item(), torch.max(torch.sum(opt_neg,dim = 1)).item())
        #     print('pos_num:', pos_num, 'neg_sum:', neg_num)

        return loss * self._scale, 0.0, 0.0, 0.0, 0.0, 0.0

        #torch.sum(dist_mat.view(-1).mul(matching_neg_flatten))  torch.max(dist_mat * is_neg)  torch.sum(opt_neg * edge_neg )

def sinkhorn_loss(C, lam, n, niter):
    """
    Given Wasserstein cost function C
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False).cuda()
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False).cuda()

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) * lam

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = 1 / lam * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = 1 / lam * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        err = (u - u1).abs().sum()

        actual_nits += 1
        if err < thresh:
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    return pi # return optimal matrix