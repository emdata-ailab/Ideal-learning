from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.kernel = nn.Parameter(torch.Tensor(98,1), requires_grad=True)
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)        

    def forward(self, inputs, targets, mode = None):

        # n = inputs.size(0)
        # # Compute similarity matrix
        # sim_mat = similarity(inputs)
        # # print(sim_mat)
        # targets = targets.cuda()
        # # split the positive and negative pairs
        # eyes_ = Variable(torch.eye(n, n)).cuda()
        # # eyes_ = Variable(torch.eye(n, n))
        # pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # neg_mask = eyes_.eq(eyes_) - pos_mask
        # pos_mask = pos_mask - eyes_.eq(1)

        # pos_sim = torch.masked_select(sim_mat, pos_mask)
        # neg_sim = torch.masked_select(sim_mat, neg_mask)

        # num_instances = len(pos_sim)//n + 1
        # num_neg_instances = n - num_instances

        # pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
        # neg_sim = neg_sim.resize(
        #     len(neg_sim) // num_neg_instances, num_neg_instances)

        # #  clear way to compute the loss first
        # loss = list()
        # c = 0

        # for i, pos_pair_ in enumerate(pos_sim):
        #     # print(i)
        #     pos_pair_ = torch.sort(pos_pair_)[0]
        #     neg_pair_ = torch.sort(neg_sim[i])[0]

        #     neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
        #     pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
        #     # pos_pair = pos_pair[1:]
        #     if len(neg_pair) < 1:
        #         c += 1
        #         continue

        #     pos_loss = torch.mean(1 - pos_pair)
        #     neg_loss = torch.mean(neg_pair)
        #     loss.append(pos_loss + neg_loss)

        # prec = float(c)/n
        # mean_neg_sim = torch.mean(neg_pair_).item()
        # mean_pos_sim = torch.mean(pos_pair_).item()
        # if len(loss) == 0:
        #     return torch.zeros([], requires_grad=True).cuda(), 0.0, 0.0, 0.0, mean_pos_sim, mean_neg_sim
        # loss = sum(loss)/n
        # return loss, 0.0, 0.0, 0.0, mean_pos_sim, mean_neg_sim


        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        eyes_=eyes_.bool()
        pos_mask = targets.expand(targets.shape[0], n).t() == targets.expand(n, targets.shape[0])
        # neg_mask = 1 - pos_mask
        neg_mask = ~pos_mask # edit yangzl
        # pos_mask[:, :n] = pos_mask[:, :n] - eyes_
        pos_mask[:, :n] = pos_mask[:, :n] & ~eyes_  # edit yangzl

        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0] 

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0] 

                select_pos_pair_idx = torch.nonzero(pos_pair_ < neg_pair_[-1] + self.margin).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                select_neg_pair_idx = torch.nonzero(neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)
        
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).cuda(), 0.0, 0.0, 0.0, mean_pos_sim, mean_neg_sim
        loss = sum(loss)/n
        return loss, 0.0, 0.0, 0.0, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(ContrastiveLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
