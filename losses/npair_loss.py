# encoding: utf-8
"""
@author:  Zhiyuan Chen
@contact: <zhiyuanchen01@gmail.com>
"""

import torch
import torch.nn.functional as F
from torch import nn

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class NpairLoss(nn.Module):
    # def __init__(self, cfg):a
    def __init__(self, args=None,  **kwargs):
        super(NpairLoss, self).__init__()
        # self._scale = cfg.MODEL.LOSSES.NPAIR.SCALE
        self._scale = args.npairscale
        self.kernel = nn.Parameter(torch.Tensor(98,1), requires_grad=True)
        self.kernel.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5) 
        
    def forward(self, embedding, targets, mode = None):
        """
        Computes the npairs loss.
        Npairs loss expects paired data where a pair is composed of samples from the
        same labels and each pairs in the minibatch have different labels. However, reid breaks 
        the rule of mini-batch due to PK-sampler when K is not 2. Our solution is to compute (K - 1) 
        npair losses for each anchor (for each anchor, there is K - 1 postive pairs), and then accumulate them.
        See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
        """

        embedding = normalize(embedding, axis=-1)
        # For distributed training, gather all features from different process.
        # if comm.get_world_size() > 1:
        #     all_embedding = concat_all_gather(embedding)
        #     all_targets = concat_all_gather(targets)
        # else:
        all_embedding = embedding
        all_targets = targets

        cos_sim = torch.matmul(embedding, all_embedding.transpose(0, 1))
        N, M = cos_sim.size()
        is_pos = ((targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()) ^ torch.eye(N))>0).cuda()
        is_neg = targets.view(N, 1).expand(N, M).ne(all_targets.view(M, 1).expand(M, N).t())

        s_p = cos_sim[is_pos].contiguous().view(N, -1)
        s_n = cos_sim[is_neg].contiguous().view(N, -1)

        # Npair: log(1 + sumexp(S_n_i - Sp)) = log(1 + exp(logsumexp(S_n_i - Sp))) = log(1 + exp(logsumexp(S_n_i) - S_p))
        loss = F.softplus((torch.logsumexp(s_n, dim = 1, keepdim=True).expand(N, M) - cos_sim)[is_pos]).mean()
        return loss * self._scale
