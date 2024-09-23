
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import time


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def l2_norm_wograd(input, axis = 1):
    input = input.cpu().numpy()
    norm = np.linalg.norm(input, ord = 2, axis = 1, keepdims=True)
    return torch.from_numpy(input / norm).cuda()
     

class SelfLoss(nn.Module):
    def __init__(self, class_num = 98, alpha=10, beta=2, margin=0.5, hard_mining=None, dim = 2048, **kwargs):
        super(SelfLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha #s
        self.beta = beta #m
        self.dim = dim
        self.hard_mining = hard_mining
        self.class_num = class_num
        self.kernel = nn.Parameter(torch.Tensor(class_num,self.dim), requires_grad=True)
       # self.kernel = nn.init.kaiming_uniform_(self.kernel, mode='fan_in')
        self.kernel.data.uniform_(-1,1).renorm_(2,0,1e-5).mul_(1e5) #   torch.sum(torch.pow(self.kernel.data.uniform_(-1,1).renorm_(2,0,1e-5).mul_(1e5), 2), 1) = [1.0, 1.0 ...]
        self.size = torch.zeros(class_num)
        #self.kernel.data = torch.from_numpy(np.load('proxy_bs16_loss30.npy')).cuda()
        
        # print('\
        # if i == 0:\n\
        #     if epoch == 0:\n\
        #         with torch.no_grad():\n\
        #             for j, data_tmp in enumerate(update_loader, 0):\n\
        #                 inputs_tmp, labels_tmp = data_tmp \n\
        #                 inputs_tmp = Variable(inputs_tmp).cuda() \n\
        #                 labels_tmp = Variable(labels_tmp).cuda() \n\
        #                 embed_feat = model(inputs_tmp) \n\
        #                 drift = criterion.center_update(embed_feat, labels_tmp) \n\
        #                 drift_deta.update(drift)')
        
        # # print('self.kernel = torch.from_numpy(np.load(\'proxy_bs16_loss30.npy\')).cuda()')

        # print('\
        #     loss, inter_, proxy_dist_ap, proxy_dist_an, sample_dist_ap, sample_dist_an = criterion(embed_feat, labels, mode) \n\
        #     loss = 1.0 * (-loss_pos[loss_mask]  + loss_neg[loss_mask] / self.alpha).sum() / max(loss_mask.sum(), 1) \n\
        #     drift = criterion.update(embed_feat, labels) \n\
        #     drift_deta.update(drift) \n\
        # ')  # yang
        

        #print('np.save(\'proxy_bs16_loss30\', criterion.kernel.data.cpu().numpy())')
                    

        

    def update(self, inputs, targets):
        drift_deta = 0.0
        for it, feature in enumerate(inputs):
            label = targets[it]
            drift_deta += torch.sum(self.kernel.data[label] * feature)
            self.kernel.data[label] = feature.detach()
        drift_deta = drift_deta / (targets.shape[0])
        return drift_deta
    
    def center_update(self, inputs, targets):
        drift_deta = 0.0
        self.kernel.data[torch.unique(targets)] = torch.zeros_like(self.kernel.data[torch.unique(targets)])
        for it, feature in enumerate(inputs):
            label = targets[it]
            drift_deta += torch.sum(self.kernel.data[label] * feature)
            self.kernel.data[label] += feature.detach()
        self.kernel.data[torch.unique(targets)] = l2_norm_wograd(self.kernel.data[torch.unique(targets)])
        drift_deta = drift_deta / (targets.shape[0])
        return drift_deta

    def forward(self, inputs, targets, mode = 'Training'):
       # print('%.8f' % torch.sum(self.kernel))
        n = inputs.size(0)
        #self.kernel.data.renorm_(2,0,1.0)
        kernel_norm = self.kernel
        #kernel_norm = l2_norm(self.kernel, axis=1) # torch.sum(torch.pow(l2_norm(self.kernel, axis=1), 2), 1)
        #import pdb; pdb.set_trace();  # torch.sum(torch.pow(self.kernel, 2), 1) torch.sum(torch.pow(inputs, 2), 1)
        kernel_norm_detach = kernel_norm.detach()
        
        proxy_sim = torch.mm(kernel_norm, torch.t(kernel_norm))
        # 
        if torch.min(targets) == 0:
            print('%.8f' % ((torch.sum(proxy_sim) - self.class_num) / (self.class_num * (self.class_num - 1)))  )        

        cos_theta = torch.mm(inputs, torch.t(kernel_norm_detach))
        #cos_theta = torch.mm(inputs, torch.t(kernel_norm))

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        label = targets.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # index = index.byte()
        index = index.bool()   # add yang      

        select_kernel = []
        for mask in index:
            select_kernel.append(kernel_norm[mask])
        select_kernel = torch.t(torch.cat(select_kernel, 0))
        phi = torch.einsum('ij, ji->i', [inputs, select_kernel]) - self.beta
        #phi = cos_theta - self.beta 
        
        output = cos_theta * 1.0
        mean_proxy_pos = output[index].sum() / index.sum()
        mean_proxy_neg = output[~index].sum() / (~index).sum()
        # mean_proxy_neg = output[1-index].sum() / (1-index).sum()
        
        output[index] = phi
        #output[index] = phi[index]  # only change the correct predicted output
        # scale up in order to make softmax work, first introduced in normface

        index = index.float()
        #loss = -50.0 * torch.sum(output*index, dim = 1) + torch.log(torch.sum(torch.exp(output * 50.0) * (1-index),dim=1))
        loss_pos = torch.sum(output*index, dim = 1) #only pos 
        #loss = torch.sum(output*index, dim = 1) + 50.0 * torch.sum(output*(1-index), dim = 1) / 97.0  #mean and mean
        #50 / 80 * torch.log(torch.sum(torch.exp(output*80.0 - 40.0)*(1-index),dim=1)).sum() / n #50 / 0.001 * torch.log((torch.exp(-output*0.001)*index).sum()) + 50 / 50.0 * torch.log(torch.sum(torch.exp(output*50)*(1-index),dim=1)).mean()
        
        loss_neg = torch.log(torch.sum(torch.exp(output * self.alpha) * (1-index),dim=1))
        margin = torch.sum(output*index,dim=1)
        loss_mask = (torch.max(output,dim=1)[0]-margin) > 0

        #loss = (self.alpha / 5.0) * torch.log(torch.sum(torch.exp(-loss_pos[loss_mask] * 5.0) ) + 1e-12) + loss_neg[loss_mask].sum() / max(loss_mask.sum(), 1)
        loss = 1.0 * (-loss_pos[loss_mask]  + loss_neg[loss_mask] / self.alpha).sum() / max(loss_mask.sum(), 1) 
        #50.0 / 10.0 * torch.log(torch.sum(torch.exp(-loss[loss_mask] * 10.0) ) + 1e-12) + loss_neg[loss_mask].sum() / max(loss_mask.sum(), 1)  #loss[loss_mask].sum() / max(loss_mask.sum(), 1)   #
        
        c = 0
        pred = torch.argmax(output,dim=1)
        prec = (pred==targets).sum().item()/n

        sim_mat = torch.mm(inputs, inputs.t())
        label_expand_1 = label.view(n, 1).repeat(1, n) 
        label_expand_2 = label.view(n, 1).repeat(n, 1) 
        sim_label = torch.eq(label_expand_1.view(n*n, -1), label_expand_2.view(n*n, -1)).float().view(n, n)
        sim_label = sim_label.byte()
        mean_pos_sim = sim_mat[sim_label].sum() / sim_label.sum()
        mean_neg_sim = sim_mat[~sim_label].sum() / (~sim_label).sum()
        # mean_neg_sim = sim_mat[1-sim_label].sum() / (1-sim_label).sum()            
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


#用真实feature替代proxy，且proxy无梯度的版本

# from __future__ import absolute_import

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable
# import numpy as np
# import time


# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#     return output

# def l2_norm_wograd(input, axis = 1):
#     input = input.cpu().numpy()
#     norm = np.linalg.norm(input, ord = 2, axis = 1, keepdims=True)
#     return torch.from_numpy(input / norm).cuda()
     

# class SelfLoss(nn.Module):
#     def __init__(self, class_num = 98, alpha=10, beta=2, margin=0.5, hard_mining=None, dim = 2048, **kwargs):
#         super(SelfLoss, self).__init__()
#         self.margin = margin
#         self.alpha = alpha #s
#         self.beta = beta #m
#         self.dim = dim
#         self.hard_mining = hard_mining
#         self.class_num = class_num
#         self.kernel = nn.Parameter(torch.Tensor(class_num,self.dim), requires_grad=True)
#        # self.kernel = nn.init.kaiming_uniform_(self.kernel, mode='fan_in')
#         self.kernel.data.uniform_(-1,1).renorm_(2,0,1e-5).mul_(1e5) #   torch.sum(torch.pow(self.kernel.data.uniform_(-1,1).renorm_(2,0,1e-5).mul_(1e5), 2), 1) = [1.0, 1.0 ...]
#         self.size = torch.zeros(class_num)
#         print('\
#             for it, feature in enumerate(inputs):\n \
#                 label = targets[it] \n\
#                 if self.size[label] == 0: \n\
#                     self.kernel.data[label] = feature \n\
#                 else: \n\
#                     self.kernel.data[label] += feature \n\
#             self.kernel.data[torch.unique(targets)] = self.kernel.data[torch.unique(targets)].renorm_(2,0,1e-5).mul(1e5) \n\
#         n = inputs.size(0) \n\
#         #self.kernel.data.renorm_(2,0,1.0) \n\
#         kernel_norm = self.kernel')
        
#     def forward(self, inputs, targets, mode = 'Training'):
#         if mode == 'Initial':
#             for it, feature in enumerate(inputs):
#                 label = targets[it]
#                 if self.size[label] == 0:
#                     self.kernel.data[label] = feature
#                 else:
#                     self.kernel.data[label] += feature
#                 #self.size[torch.unique(targets)] += targets.shape[0] * 1.0 / torch.unique(targets).shape[0] 
            
#             #self.kernel.data[torch.unique(targets)] = self.kernel.data[torch.unique(targets)] / (targets.shape[0] * 1.0 / torch.unique(targets).shape[0])
#             #self.kernel.data[torch.unique(targets)] = self.kernel.data[torch.unique(targets)].renorm_(2,0,1.0)
#             self.kernel.data[torch.unique(targets)] = self.kernel.data[torch.unique(targets)].renorm_(2,0,1e-5).mul(1e5)
#             #self.kernel.data[torch.unique(targets)] = l2_norm_wograd(self.kernel.data[torch.unique(targets)])

#        # print('%.8f' % torch.sum(self.kernel))
#         n = inputs.size(0)
#         #self.kernel.data.renorm_(2,0,1.0)
#         kernel_norm = self.kernel
#         #kernel_norm = l2_norm(self.kernel, axis=1) # torch.sum(torch.pow(l2_norm(self.kernel, axis=1), 2), 1)
#         #import pdb; pdb.set_trace();  # torch.sum(torch.pow(self.kernel, 2), 1) torch.sum(torch.pow(inputs, 2), 1)
#         kernel_norm_detach = kernel_norm.detach()
        
#         proxy_sim = torch.mm(kernel_norm, torch.t(kernel_norm))
#         # 
#         if torch.min(targets) == 0:
#             print('%.8f' % ((torch.sum(proxy_sim) - self.class_num) / (self.class_num * (self.class_num - 1)))  )        

#         cos_theta = torch.mm(inputs, torch.t(kernel_norm_detach))
#         #cos_theta = torch.mm(inputs, torch.t(kernel_norm))

#         cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

#         label = targets.view(-1, 1)  # size=(B,1)
#         index = cos_theta.data * 0.0  # size=(B,Classnum)
#         index.scatter_(1, label.data.view(-1, 1), 1)
#         index = index.byte()        

#         select_kernel = []
#         for mask in index:
#             select_kernel.append(kernel_norm[mask])
#         select_kernel = torch.t(torch.cat(select_kernel, 0))
#         phi = torch.einsum('ij, ji->i', [inputs, select_kernel]) - self.beta
#         #phi = cos_theta - self.beta 
        
#         output = cos_theta * 1.0
#         mean_proxy_pos = output[index].sum() / index.sum()
#         mean_proxy_neg = output[1-index].sum() / (1-index).sum()
        
#         output[index] = phi
#         #output[index] = phi[index]  # only change the correct predicted output
#         # scale up in order to make softmax work, first introduced in normface

#         index = index.float()
#         #loss = -50.0 * torch.sum(output*index, dim = 1) + torch.log(torch.sum(torch.exp(output * 50.0) * (1-index),dim=1))
#         loss_pos = torch.sum(output*index, dim = 1) #only pos 
#         #loss = torch.sum(output*index, dim = 1) + 50.0 * torch.sum(output*(1-index), dim = 1) / 97.0  #mean and mean
#         #50 / 80 * torch.log(torch.sum(torch.exp(output*80.0 - 40.0)*(1-index),dim=1)).sum() / n #50 / 0.001 * torch.log((torch.exp(-output*0.001)*index).sum()) + 50 / 50.0 * torch.log(torch.sum(torch.exp(output*50)*(1-index),dim=1)).mean()
        
#         loss_neg = torch.log(torch.sum(torch.exp(output * self.alpha) * (1-index),dim=1))
#         margin = torch.sum(output*index,dim=1)
#         loss_mask = (torch.max(output,dim=1)[0]-margin) > 0

#         #loss = (self.alpha / 5.0) * torch.log(torch.sum(torch.exp(-loss_pos[loss_mask] * 5.0) ) + 1e-12) + loss_neg[loss_mask].sum() / max(loss_mask.sum(), 1)
#         loss =  (-loss_pos[loss_mask] * self.alpha + loss_neg[loss_mask]).sum() / max(loss_mask.sum(), 1)
#         #50.0 / 10.0 * torch.log(torch.sum(torch.exp(-loss[loss_mask] * 10.0) ) + 1e-12) + loss_neg[loss_mask].sum() / max(loss_mask.sum(), 1)  #loss[loss_mask].sum() / max(loss_mask.sum(), 1)   #
        
#         c = 0
#         pred = torch.argmax(output,dim=1)
#         prec = (pred==targets).sum().item()/n

#         sim_mat = torch.mm(inputs, inputs.t())
#         label_expand_1 = label.view(n, 1).repeat(1, n) 
#         label_expand_2 = label.view(n, 1).repeat(n, 1) 
#         sim_label = torch.eq(label_expand_1.view(n*n, -1), label_expand_2.view(n*n, -1)).float().view(n, n)
#         sim_label = sim_label.byte()
#         mean_pos_sim = sim_mat[sim_label].sum() / sim_label.sum()
#         mean_neg_sim = sim_mat[1-sim_label].sum() / (1-sim_label).sum()            
#         return loss, prec, mean_proxy_pos, mean_proxy_neg, mean_pos_sim, mean_neg_sim


# def main():
#     data_size = 32
#     input_dim = 3
#     output_dim = 512
#     num_class = 4
#     # margin = 0.5
#     x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
#     # print(x)
#     w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
#     inputs = x.mm(w)
#     y_ = 8 * list(range(4))
#     targets = Variable(torch.LongTensor(y_))

#     print(SelfLoss()(inputs, targets))


# if __name__ == '__main__':
#     main()
#     print('Congratulations to you!')

