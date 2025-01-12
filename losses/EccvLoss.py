from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class EccvLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(EccvLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha #s
        self.beta = beta #m
        self.hard_mining = hard_mining
        self.kernel = nn.Parameter(torch.Tensor(98,2048), requires_grad=True)
       # self.kernel = nn.init.kaiming_uniform_(self.kernel, mode='fan_in')
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)

    def forward(self, inputs, targets, mode=None):
        n = inputs.size(0)
        # targets = targets
        kernel_norm = l2_norm(self.kernel, axis=1)
        cos_theta = torch.mm(inputs, torch.t(kernel_norm))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.beta
        label = targets.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        
        mean_proxy_pos = output[index].sum() / index.sum()
        mean_proxy_neg = output[1-index].sum() / (1-index).sum()
        
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.alpha  # scale up in order to make softmax work, first introduced in normface
        # print(output.shape)
        

        index = index.float()
        logit = torch.log(torch.sum(torch.exp(output)*(1-index),dim=1))
        margin = torch.sum(output*index,dim=1)
        loss = logit - margin

        loss_mask = (torch.max(output,dim=1)[0]-margin )>0

        loss = loss[loss_mask].sum()/n

        c = 0
        pred = torch.argmax(output,dim=1)
        prec = (pred==targets).sum().item()/n

        
        sim_mat = torch.mm(inputs, inputs.t())
        label_expand_1 = label.view(n, 1).repeat(1, n) 
        label_expand_2 = label.view(n, 1).repeat(n, 1) 
        sim_label = torch.eq(label_expand_1.view(n*n, -1), label_expand_2.view(n*n, -1)).float().view(n, n)
        sim_label = sim_label.byte()
        mean_pos_sim = sim_mat[sim_label].sum() / sim_label.sum()
        mean_neg_sim = sim_mat[1-sim_label].sum() / (1-sim_label).sum()
        
        return loss, prec, mean_proxy_pos, mean_proxy_neg, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 512
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(4))
    targets = Variable(torch.LongTensor(y_))

    print(SelfLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


