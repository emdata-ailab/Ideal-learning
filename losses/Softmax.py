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

class SoftmaxLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(SoftmaxLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha #s
        self.beta = beta #m
        self.hard_mining = hard_mining
        self.kernel = nn.Parameter(torch.Tensor(512,98))
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)




    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        # targets = targets

        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(inputs, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.beta
        label = targets.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.alpha  # scale up in order to make softmax work, first introduced in normface
        # print(output.shape)
        loss = F.cross_entropy(output,targets)

        c = 0

        pred = torch.argmax(output,dim=1)

        pos_sim,neg_sim = [],[]

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            mean_neg_sim_ = torch.mean(neg_pair_).item()
            mean_pos_sim_ = torch.mean(pos_pair_).item()

            pos_sim.append(mean_pos_sim_)
            neg_sim.append(mean_neg_sim_)


        # loss = sum(loss) / n
        # print('pred:',pred.dtype,'target:',targets.dtype)
        prec = (pred==targets).sum().item()/n
        # print(prec)
        mean_pos_sim = sum(pos_sim)/n
        mean_neg_sim = sum(neg_sim)/n

        return loss, prec, mean_pos_sim, mean_neg_sim


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

    print(SoftmaxLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


