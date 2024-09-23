# coding=utf-8
from __future__ import print_function, absolute_import
import time, pdb, random
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np
from tqdm import tqdm
cudnn.benchmark = True

def train(epoch, model, criterion, optimizer, train_loader, update_loader, args, mode = 'Training', sample_count=0):
    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    proxy_pos_sims = AverageMeter()
    proxy_neg_sims = AverageMeter()
    sample_pos_sims = AverageMeter()
    sample_neg_sims = AverageMeter()
    drift_deta =  AverageMeter()
    feature_Mask = None
    end = time.time()
    transform_index,index = None,None
    length = len(train_loader)
    freq = min(args.print_freq, length)
    train_loader.dataset.sample_count = sample_count
    # train_iter = tqdm(enumerate(train_loader, 0), total=len(train_loader))
    for i, data_src in enumerate(train_loader, 0):
        # if i == 0:
        #     if epoch == 0:
        #         with torch.no_grad():
        #             for j, data_tmp in enumerate(update_loader, 0):
        #                 inputs_tmp, labels_tmp = data_tmp
        #                 # wrap them in Variable
        #                 inputs_tmp = Variable(inputs_tmp).cuda()
        #                 labels_tmp = Variable(labels_tmp).cuda()
        #                 embed_feat = model(inputs_tmp)
        #                 drift = criterion.center_update(embed_feat, labels_tmp)
        #                 drift_deta.update(drift)
        
        if args.loader_type == '':
            inputs, labels = data_src
        else:
            inputs, labels, transform_indexs, indexs = data_src
            
            if 'repeat' in args.loader_type:
                inputs = torch.cat(inputs.chunk(4,1),dim=0)
                # inputs = inputs.chunk(4,1)[2].repeat(4,1,1,1)
                labels = labels.repeat(4)

        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()

        experiment = args.experiment_type  # cat-div2, 1normal, 2normal
        normalized = experiment != "1normal"

        embed_feat = model(inputs)
        # import pdb; pdb.set_trace()
        if args.head_type != "":
            if feature_Mask is None:
                feature_Mask = create_fmask(embed_feat)
            if feature_Mask.shape != embed_feat.shape:
                feature_Mask = create_fmask(embed_feat)
            embed_feat = embed_feat * feature_Mask
        if 'multi' in args.loader_type and not experiment:
            labels_list = labels.chunk(4,0)
            embed_feat = embed_feat.chunk(4,0)

        if experiment == "cat-div2":  # normalized True
            embed_feat = torch.cat(embed_feat.chunk(4,0), dim=1)/2 
            labels = labels.chunk(4,0)[0]
        if experiment == "1normal" or experiment == "2normal" :  # {1normal:normalized=False, 2normal:normalized=True}
            embed_feat = torch.cat(embed_feat.chunk(4,0), dim=1)
            norm = embed_feat.norm(dim=1, p=2, keepdim=True)
            embed_feat = embed_feat.div(norm.expand_as(embed_feat))
            labels = labels.chunk(4,0)[0]

        '''
        if optimizer.param_groups[2]['lr'] == 0.0:
            sublabel = labels[::args.num_instances]
            criterion.kernel.data[sublabel] = criterion.kernel.data[sublabel] * epoch / (epoch + 1)\
            + torch.mean(embed_feat.view(-1, args.num_instances, args.dim), 1) / (epoch + 1)
        '''
        if args.loss == 'Mvp':
            proxy_dist_ap = criterion(embed_feat, labels, mode = 'pos')
            proxy_dist_an = criterion(embed_feat, labels, mode = 'neg')
            loss = proxy_dist_ap + proxy_dist_an
            inter_ = 0
            sample_dist_ap = 0
            sample_dist_an = 0
        else:
            if 'multi' not in args.loader_type or experiment != "":
                loss, inter_, proxy_dist_ap, proxy_dist_an, sample_dist_ap, sample_dist_an = criterion(embed_feat, labels, mode)
            else:
                criterion_rst = [criterion(embed_feat_, labels_, mode) for embed_feat_, labels_ in zip(embed_feat, labels_list)]

                if args.loss_balance:
                    loss = sum([criterion_[0] for criterion_ in criterion_rst]) / 4
                else:
                    loss = criterion_rst[0][0]/2 + sum([criterion_[0] for criterion_ in criterion_rst[1:]]) / 6
                inter_ = sum([criterion_[1] for criterion_ in criterion_rst]) / 4
                proxy_dist_ap = sum([criterion_[2] for criterion_ in criterion_rst]) / 4
                proxy_dist_an = sum([criterion_[3] for criterion_ in criterion_rst]) / 4
                sample_dist_ap = sum([criterion_[4] for criterion_ in criterion_rst]) / 4
                sample_dist_an = sum([criterion_[5] for criterion_ in criterion_rst]) / 4
                # pdb.set_trace()
            # drift = criterion.update(embed_feat, labels)
            # drift_deta.update(drift)
            #np.save('proxy_bs16_loss30', criterion.kernel.data.cpu().numpy())

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)
        
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        accuracy.update(inter_)
        proxy_pos_sims.update(proxy_dist_ap)
        proxy_neg_sims.update(proxy_dist_an)
        sample_pos_sims.update(sample_dist_ap)
        sample_neg_sims.update(sample_dist_an)

        # print(accuracy.avg)

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            # train_iter.set_description('Epoch:[{0:03d}], Time {batch_time.avg:.2f}, Loss {loss.avg:.3f}, Accuracy {accuracy.avg:.3f}, proxP {proxpos.avg:.3f}, proxN {proxneg.avg:.3f}, realP {pos.avg:.3f}, realN {neg.avg:.3f}, drift {drift.avg:.3f}\t'.format
            #       (epoch + 1, batch_time=batch_time,
            #        loss=losses, accuracy=accuracy, proxpos=proxy_pos_sims, proxneg=proxy_neg_sims, pos=sample_pos_sims, neg=sample_neg_sims, drift=drift_deta))
            print('Epoch:[{0:03d}], Loss {loss.avg:.3f}, Accuracy {accuracy.avg:.3f}'.format(epoch + 1, loss=losses, accuracy=accuracy))

        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
        
    sample_count = sample_count + i + 1
    return losses.avg, sample_count

def create_fmask(embed_feat):
    outmask = torch.zeros_like(embed_feat)
    h, w = outmask.shape
    for h_st, w_st in zip(range(0, h, int(h/4)), range(0, w, int(w/4))):
        outmask[h_st:h_st+int(h/4), w_st:w_st+int(w/4)] = 1
    return outmask
