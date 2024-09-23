# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os, pdb
import sys
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import torchvision
from tensorboardX import SummaryWriter
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint
from trainer import train
from tester import test, test_multi_queryloader
from utils import orth_reg
from utils.osutils import str2bool

import DataSet
import numpy as np
import os.path as osp
cudnn.benchmark = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

use_gpu = True

# Batch Norm Freezer : bring 2% improvement on CUB 
def set_bn_eval(m): #意义在于不仅仅让可学习仿射变换的mean和sigma不变化，也让滑动平均的mean和sigma不变化，且训练的时候不是通过mini-batch的mean和sigma来norm，而是用固定的滑动平均来norm，再经过固定的仿射变换，让训练和测试没有区别了。
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
   # elif classname.find('BatchNorm') != -1:
   #     if m.affine:
   #         nn.init.constant_(m.weight, 1.0)
   #         nn.init.constant_(m.bias, 0.0)
        
class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, out_dim):
        super(FineTuneModel, self).__init__()
        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True)
            )
            self.pooling_output = nn.Linear(4096, num_classes)
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.layernorm = nn.Sequential(
                nn.LayerNorm(2048, elementwise_affine = False)
            )
            self.bnnorm = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
               # nn.ReLU(inplace=True)
            )
            self.bnnorm.apply(weights_init_kaiming)
            self.classifier = nn.Sequential(
                nn.Linear(2048, out_dim)
            )
            self.modelName = 'resnet'  
        elif arch.startswith('inception') :
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, out_dim)
            )
            self.modelName = 'inception'  
        else :
            raise("Finetuning not supported on this architecture yet")

        # # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        
        if self.modelName == 'alexnet' :
            x = x.view(x.size(0), 256 * 6 * 6)
        elif self.modelName == 'resnet' :
            x = x.view(x.size(0), -1)
       # x = self.layernorm(x)
       # x = self.classifier(x) # + torch.tensor(5.0).cuda()
       # norm = x.norm(dim=1, p=2, keepdim=True)
       # x = x.div(norm.expand_as(x))
        return x

    
def main(args):
    # pdb.set_trace()
    save_dir = args.save_dir
    args.multi_test = str2bool(args.multi_test)
    args.debug = str2bool(args.debug)
    pretrained = str2bool(args.pretrained)
    args.loss_balance = str2bool(args.loss_balance)
    log_name = args.log_name
    ckpt_dir = os.path.join(save_dir, args.log_name)
    log_timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S')
    print(log_name)
    mkdir_if_missing(save_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    sys.stdout = logging.Logger(os.path.join(ckpt_dir, log_timestamp))
    print('==========\nArgs:{}\n=========='.format(args))
    display(args)
    
    start = 0
    # create model
    model = models.create(args.net, pretrained=pretrained, dim=args.dim, head_type=args.head_type)

    checkpoint_dir = osp.join(args.save_dir, "checkpoints", args.log_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # pdb.set_trace()
    writer = SummaryWriter(log_dir=ckpt_dir)
    # mine
    # print("=> using pre-trained model '{}'".format(args.net))
    # model = torchvision.models.__dict__[args.net](pretrained=True)
    # model = FineTuneModel(model, args.net, args.dim)
   
    # for vgg and densenet
    # import pdb; pdb.set_trace()
    if args.resume is None:
        model_dict = model.state_dict()

    else:
        # resume model
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        model.load_state_dict(weight)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    # model.apply(set_bn_eval) #############
    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')
        
    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    # import pdb; pdb.set_trace()
    if not isinstance(model.module.classifier, list):
        new_param_ids = set(map(id, model.module.classifier.parameters()))
    else:
        new_param_ids = set.union(set(map(id, model.module.classifier[0].parameters())),set(map(id, model.module.classifier[1].parameters())),set(map(id, model.module.classifier[2].parameters())),set(map(id, model.module.classifier[3].parameters())),)

    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.module.parameters() if
                   id(p) not in new_param_ids] 
    
    criterion = losses.create(args.loss, class_num = args.class_num, margin=args.margin, alpha=args.alpha,beta=args.beta, base=args.loss_base,hard_mining=True, dim = args.dim, norm = args.feature_norm, args=args).cuda()
    param_groups = [
                {'params': base_params, 'lr': args.lr},
                {'params': new_params, 'lr': args.lr},
                {'params': criterion.kernel, 'lr': args.lr}]

    print('initial model is save at %s' % save_dir)

    if args.optim == 'ADAM':
        print("using adam optim")
        print("weight_decay is ", args.weight_decay)        
        optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        print("using s-128.gd optim, momentum is ",  args.momentum)
        print("weight_decay is ", args.weight_decay)       
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum = args.momentum,
                                     weight_decay=args.weight_decay)    
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    # Decor_loss = losses.create('decor').cuda()
    if args.loader_type=='':
        data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root,train_trans=args.train_trans,test_trans=args.test_trans,batch_size=args.batch_size, debug=args.debug)
    else:
        data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root,train_trans=args.train_trans,test_trans=args.test_trans,loader_type=args.loader_type,batch_size=args.batch_size)
    
    dataloader_batchSize = args.batch_size if 'repeat' not in args.loader_type else int(args.batch_size/4)
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=dataloader_batchSize,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)

    update_loader = torch.utils.data.DataLoader(
        data.train, batch_size=dataloader_batchSize,
        sampler=FastRandomIdentitySampler(data.train, num_instances=1),
        drop_last=False, pin_memory=True, num_workers=args.nThreads)

    if args.data in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=dataloader_batchSize, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=args.nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=dataloader_batchSize,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.nThreads)

    else:
        query_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=dataloader_batchSize,
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.nThreads)
        gallery_loader = query_loader
        if args.multi_test:
            query_dict = {}
            for key, query_dataloader in data.query_dict.items():
                query_dict[key] = torch.utils.data.DataLoader(query_dataloader, batch_size=dataloader_batchSize,shuffle=False, drop_last=False, pin_memory=True, num_workers=args.nThreads)

    # save the train information
    lst_max_recall = -1
    pos = -1
    decrease_time = 0
    sample_count = 0
    for epoch in range(start, args.epochs):        
        if epoch == 0:
            pos = epoch
            optimizer.param_groups[0]['lr'] = args.lr * args.basemodel_lr_mul
            optimizer.param_groups[1]['lr'] = args.lr
            optimizer.param_groups[2]['lr'] = args.lr

        #optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr'] = 0
        #optimizer.param_groups[2]['lr'] = 0
        #print('now lr is:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'])
        #train(epoch=epoch, model=model, criterion=criterion,
        #    optimizer=optimizer, train_loader=train_loader, args=args, mode = 'Initial')
        optimizer.param_groups[0]['lr'] = args.lr * args.basemodel_lr_mul
        optimizer.param_groups[1]['lr'] = args.lr
        optimizer.param_groups[2]['lr'] = 0.0   
        print('now lr is:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'])
        loss, sample_count =train(epoch=epoch, model=model, criterion=criterion,
            optimizer=optimizer, train_loader=train_loader, update_loader=update_loader, args=args, sample_count=sample_count)  

        writer.add_scalar('train/loss', loss, epoch)     
        torch.cuda.empty_cache()

        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(checkpoint_dir, 'Epoch' + str(epoch + 1) + '.pth.tar'))
            
            tolerance = [100, 100, 100, 2, 2, 2] #[5, 5, 5, 15, 15, 15, 15, 15]
            if not args.multi_test:
                recall_1 = test(epoch=epoch, model=model, query_loader=query_loader, gallery_loader = gallery_loader, args=args)
                writer.add_scalar('test/recall', recall_1, epoch)

            else:
                recall_dict = test_multi_queryloader(epoch=epoch, model=model, query_dict=query_dict, args=args, head_type=args.head_type)
                for key, recall_ in recall_dict.items():
                    writer.add_scalar('test/recall_{}'.format(key), recall_, epoch)
                recall_1 = recall_dict['catFeature']
                
            model.train()
            if args.freeze_BN is True:
                model.apply(set_bn_eval)
            
            if recall_1 > lst_max_recall:
                lst_max_recall = recall_1
                save_checkpoint({'state_dict': state_dict, 'epoch': (epoch+1)}, 
                is_best=True, fpath=osp.join(checkpoint_dir, "Best-epoch"+str(epoch + 1) + '.pth.tar'))
                pos = epoch
            elif epoch - pos >= args.save_step * tolerance[decrease_time] : 
                args.lr = args.lr / 3.0
                optimizer.param_groups[0]['lr'] = args.lr  * args.basemodel_lr_mul
                optimizer.param_groups[1]['lr'] = args.lr
                optimizer.param_groups[2]['lr'] = args.lr
                print('now lr is:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'])
                pos = epoch
                lst_max_recall = recall_1
                decrease_time += 1                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--feature-norm', action='store_true', help='feature norm')
    parser.add_argument('--log_name', type=str, default='train',help='none')
    parser.add_argument('--debug', type=str, default='False',help='none')
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')

    parser.add_argument('--alpha', default=30.0, type=float, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # ----------------------start of optLoss---------------------
    parser.add_argument('--opt_marginlb', default=0.4, type=float, help='不同类的余弦相似度的最小值')
    parser.add_argument('--opt_marginub', default=0.6, type=float, help='同类的余弦相似度最大值')
    parser.add_argument('--opt_scale', default=1.0, type=float, help='weight of optLoss')
    
    parser.add_argument('--npairscale', default=1.0, type=float, help='weight of optLoss')
    # ----------------------end of optLoss-----------------------

    # network
    parser.add_argument('--optim', default='SGD', required=True,
                        help='optim for training network')
    parser.add_argument('--freeze_BN', action='store_true', help='Freeze BN')
    parser.add_argument('--multi_test', type=str, default='False',help='Freeze BN')
    parser.add_argument('--head_type', type=str, default='',help='type of multi head, multi, multi_split')  
    parser.add_argument('--loss_balance', type=str, default='True',help='True, False') 
    
    parser.add_argument('--data', default='cub', required=True,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None, help='path to Data Set')
    parser.add_argument('--loader_type', type=str, default='', help='loader_type: batch_single_angle, batch_multi_angle')
    parser.add_argument('--experiment_type', type=str, default='', help='cat-div2, 1normal, 2normal')

    parser.add_argument('--net', default='VGG16-BN')
    parser.add_argument('--loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')
    parser.add_argument('--class_num', default=98, type=int,
                        help='the number of categories in training set')

    # Resume from checkpoint
    parser.add_argument('--resume', '-r', default=None,
                        help='the path of the pre-trained model')
    parser.add_argument('--pretrained', default="True", type=str,help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=1, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--basemodel-lr-mul', type=float, default=0.1)

    parser.add_argument('--loss_base', type=float, default=0.75)

    parser.add_argument('--train_trans', type=str, default='rand-crop', help='train dataset transform type')
    parser.add_argument('--test_trans', type=str, default='center-crop', help='test dataset transform type')
    main(parser.parse_args())
