from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class SelfBatchLoss(nn.Module):
    def __init__(self, alpha=40, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(SelfBatchLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta


    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            pos_pair = pos_pair_
            neg_pair = neg_pair_

            if len(neg_pair) < 1 or len(pos_pair) < 1  :
                continue
            if pos_pair[0] - self.beta > neg_pair[-1]:
                # c+=1
                continue
            pos_loss = torch.log(torch.sum(torch.exp(-self.alpha * (pos_pair-self.beta))))
            neg_loss = torch.log(torch.sum(torch.exp(0.1*self.alpha * (neg_pair))))

            debug_loss = neg_pair[-1] - pos_pair[0] + self.beta

            c+= debug_loss.item()


            loss.append(pos_loss + neg_loss)
        loss = sum(loss) / n
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8 * list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(SelfBatchLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

