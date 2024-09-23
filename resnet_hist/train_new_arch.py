import torch, math, time, argparse, os, random
import torch.nn as nn
import numpy as np

from torch.utils.data.sampler import BatchSampler
import dataset
from dataset.sampler import FastRandomIdentitySampler
from dataset.Inshop import Inshop_Dataset

import utils
from hist import *

import net
from net.resnet_new_arch import *

from tqdm import *
import wandb
import pdb

from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--gpu-id',
                    default=1,
                    type=int,
                    help='ID of GPU that is used for training.'
                    )

parser.add_argument('--workers',
                    default=4,
                    type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )

parser.add_argument('--data',
                    default='car',
                    help='Training dataset, e.g. cub, cars, SOP'
                    )

parser.add_argument('--data_root', 
                    type=str, 
                    default='/dataset/image_retrieval/', 
                    help='path to Data Set')

parser.add_argument('--model',
                    default='resnet50',
                    # default='bn_inception',
                    help='Model for training, e.g. bn_inception, resnet50'
                    )
parser.add_argument('--num_instances', 
                    default=4, 
                    type=int, 
                    metavar='n',                        
                    help=' number of samples from one class in mini-batch'
                    )

parser.add_argument('--embedding-size',
                    default=512,
                    type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.'
                    )

parser.add_argument('--hgnn-hidden',
                    default=512,
                    type=int,
                    help='Size of hidden units in HGNN.'
                    )

parser.add_argument('--add-gmp',
                    default=1,
                    type=int,
                    help='if 1, add GMP feature, else if set to 0, only use GAP.'
                    )

parser.add_argument('--batch-size',
                    default=32,
                    type=int,
                    help='Number of samples per batch.'
                    )

parser.add_argument('--epochs',
                    # default=50,
                    default=500,
                    type=int,
                    dest='nb_epochs',
                    help='Number of training epochs.'
                    )

parser.add_argument('--optimizer',
                    default='adam',
                    help='Optimizer setting'
                    )

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='Learning rate for embedding network parameters'
                    )

parser.add_argument('--lr-ds',
                    default=1e-1,
                    type=float,
                    help='Learning rate for class prototypical distribution parameters'
                    )

parser.add_argument('--lr-hgnn-factor',
                    default=10,
                    type=float,
                    help='Learning rate multiplication factor for HGNN parameters'
                    )

parser.add_argument('--weight-decay',
                    default=5e-4,
                    type=float,
                    help='Weight decay setting'
                    )

parser.add_argument('--lr-decay-step',
                    default=10,
                    type=int,
                    help='Learning decay step setting'
                    )

parser.add_argument('--lr-decay-gamma',
                    default=0.5,
                    type=float,
                    help='Learning decay gamma setting'
                    )

parser.add_argument('--tau',
                    default=32,
                    type=float,
                    help='temperature scale parameter for softmax'
                    )

parser.add_argument('--alpha',
                    default=0.9,
                    type=float,
                    help='hardness scale parameter for construction of H'
                    )

parser.add_argument('--ls',
                    default=1,
                    type=float,
                    help='loss scale balancing parameters (lambda_s)'
                    )

parser.add_argument('--IPC',
                    default=0,
                    type=int,
                    help='Balanced sampling, images per class'
                    )

parser.add_argument('--warm',
                    default=1,
                    type=int,
                    help='Warmup training epochs, if set to 0, do not warm up'
                    )

parser.add_argument('--bn-freeze',
                    default=1,
                    type=int,
                    help='Batch normalization parameter freeze, if set to 0, do not freeze'
                    )

parser.add_argument('--layer-norm',
                    default=1,
                    type=int,
                    help='Layer normalization'
                    )

parser.add_argument('--remark',
                    default='',
                    help='Any remark'
                    )

parser.add_argument('--run-num',
                    default=1,
                    type=int,
                    help='The number of repetitive run'
                    )

parser.add_argument('--head_type', 
                    type=str, 
                    default='',
                    help='type of multi head, multi, multi_split'
                    )

parser.add_argument('--loader_type', 
                    type=str, 
                    default='', 
                    help='loader_type: batch_single_angle, batch_multi_angle'
                    )

parser.add_argument('--experiment_type', 
                    type=str, 
                    default='', 
                    help='cat-div2, 1normal, 2normal'
                    )

parser.add_argument('--train_trans', 
                    type=str, 
                    default='hist_train', 
                    help='train dataset transform type'
                    )
parser.add_argument('--test_trans', 
                    type=str, 
                    default='hist_test', 
                    help='test dataset transform type'
                    )

parser.add_argument('--multi_test', 
                    type=str, 
                    default='False',
                    help='Freeze BN'
                    )

parser.add_argument('--save_dir', 
                    default='',
                    help='where the trained models save'
                    )

args = parser.parse_args()


# Set fixed random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


torch.cuda.set_device(args.gpu_id)

# Directory for Logging
log_folder_name = 'run_{}'.format(args.run_num)
LOG_DIR = '/workspace/hist_ideal_resnet50/code/script/stage1.1/res/{}/{}'.format(args.save_dir, log_folder_name)

if not os.path.exists('{}'.format(LOG_DIR)):
    os.makedirs('{}'.format(LOG_DIR))


# Wandb Initialization
wb_project_name = 'hist_{}'.format(args.data)
wandb.init(project=wb_project_name, notes=LOG_DIR)
wandb.config.update(args)
wandb.run.name = '{}'.format(log_folder_name)
 

dataloader_batchSize = args.batch_size if 'repeat' not in args.loader_type else int(args.batch_size/4)
if args.loader_type=='':
    data = dataset.create(args.data, root=args.data_root, train_trans=args.train_trans, test_trans=args.test_trans, batch_size=args.batch_size)
else:
    data = dataset.create(args.data, root=args.data_root, train_trans=args.train_trans, test_trans=args.test_trans, loader_type=args.loader_type, batch_size=args.batch_size)
# Dataset Loader and Sampler
dl_tr = torch.utils.data.DataLoader(
    data.train, 
    batch_size=dataloader_batchSize,
    shuffle=True,
    drop_last=True, 
    pin_memory=True, 
    num_workers=args.nb_workers
    )

if args.data in ['shop', 'jd_test']:
    dl_gallery = torch.utils.data.DataLoader(
        data.gallery, 
        batch_size=dataloader_batchSize, 
        shuffle=False,
        drop_last=False, 
        pin_memory=True, 
        num_workers=args.nb_workers
        )

    dl_query = torch.utils.data.DataLoader(
        data.query, 
        batch_size=dataloader_batchSize,
        shuffle=False, 
        drop_last=False,
        pin_memory=True, 
        num_workers=args.nb_workers
        )

else:
    dl_query = torch.utils.data.DataLoader(
        data.gallery, 
        batch_size=dataloader_batchSize,
        shuffle=False, 
        drop_last=False, 
        pin_memory=True,
        num_workers=args.nb_workers
        )

    dl_gallery = dl_query
    if args.multi_test: # 如果不适用multi_test，不走这一项
        query_dict = {}
        for key, query_dataloader in data.query_dict.items():
            query_dict[key] = torch.utils.data.DataLoader(
                query_dataloader, 
                batch_size=dataloader_batchSize,
                shuffle=False, 
                drop_last=False, 
                pin_memory=True, 
                num_workers=args.nb_workers
            )

nb_classes=data.nb_classes()

# Feature Embedding (Backbone)

model = resnet50(embedding_size=args.sz_embedding,
                pretrained=True,
                is_norm=args.layer_norm,
                bn_freeze=args.bn_freeze,
                head_type='',
                )

model = model.cuda()

# Class Prototypical Distributions -> Hypergraph model
d2hg = CDs2Hg(nb_classes=nb_classes,
              sz_embed=args.sz_embedding,
              tau=args.tau,
              alpha=args.alpha)
d2hg.cuda()


# Hypergraph Neural Network
hnmp = HGNN(nb_classes=nb_classes,
            sz_embed=args.sz_embedding,
            hidden=args.hgnn_hidden)
hnmp.cuda()


# Overall train parameters
param_groups = []
param_groups.append({'params': list(set(model.parameters()).difference(set(model.classifier.embedding.parameters()))),
                     'lr': args.lr})
param_groups.append({'params': model.classifier.embedding.parameters(),
                     'lr': args.lr})
param_groups.append({'params': d2hg.parameters(),
                     'lr': args.lr_ds})
param_groups.append({'params': hnmp.parameters(),
                     'lr': args.lr * args.lr_hgnn_factor})


# Optimizer Setting
if args.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups,
                          lr=float(args.lr),
                          weight_decay=args.weight_decay,
                          momentum=0.9,
                          nesterov=True
                          )

elif args.optimizer == 'adam':
    opt = torch.optim.Adam(param_groups,
                           lr=float(args.lr),
                           weight_decay=args.weight_decay
                           )

elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups,
                              lr=float(args.lr),
                              alpha=0.9,
                              weight_decay=args.weight_decay,
                              momentum=0.9
                              )

elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups,
                            lr=float(args.lr),
                            weight_decay=args.weight_decay
                            )

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


print("Training arguments: {}".format(vars(args)))
print("===============================")

losses_list = []
best_recall = [0]
best_epoch = 0
break_out_flag = False

""" Warm up: Train only new params, helps stabilize learning. """
if args.warm > 0:
    print("** Warm up training for {} epochs... **".format(args.warm))
    freeze_params = param_groups[0]['params']

    for epoch in range(0, args.warm):

        model.train()
        losses_per_epoch = []

        # BN freeze
        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Freeze backbone params
        for param in freeze_params:
            param.requires_grad = False

        pbar = tqdm(enumerate(dl_tr))

        for batch_idx, (x, y) in pbar:
            x = x.cuda()
            y = y.cuda()-1
            
            m = model(x.squeeze())
            targets = y.squeeze()

            # Hypergraph construction & distribution loss
            dist_loss, H = d2hg(m, targets)
            H.cuda()

            # Hypergraph node classification
            out = hnmp(m, H)
            criterion = nn.CrossEntropyLoss()
            # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            ce_loss = criterion(out, targets)


            loss = dist_loss + args.ls * ce_loss

            opt.zero_grad()
            loss.backward()

            opt.step()

            pbar.set_description(
                'Warm-up Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(dl_tr),
                           100. * batch_idx / len(dl_tr),
                    loss.item()))

            losses_per_epoch.append(loss.data.cpu().numpy())

            if np.isnan(losses_per_epoch[-1]):
                break_out_flag = True
                break

        if break_out_flag:
            print("** Failed training (NaN Loss)... **")
            break

        if epoch >= 0:
            with torch.no_grad():
                print("** Evaluating... **")
                if args.data == 'Inshop':
                    Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                elif args.data != 'SOP':
                    Recalls, NMIs = utils.evaluate_cos(model, dl_query, eval_nmi=False)
                else:
                    Recalls, NMIs = utils.evaluate_cos_SOP(model, dl_query, eval_nmi=False)

    # Unfreeze backbone params
    for param in freeze_params:
        param.requires_grad = True

    print("** Warm up training done... **")


print("===============================")
print("** Training for {} epochs... **".format(args.nb_epochs))
for epoch in range(0, args.nb_epochs):

    model.train()
    losses_per_epoch = []

    # BN freeze
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.modules()
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        
        x = x.cuda()
        y = y.cuda()-1

        m = model(x.squeeze())
        target = y.squeeze()

        # Hypergraph construction & distribution loss
        dist_loss, H = d2hg(m, target)
        H.cuda()

        # Hypergraph node classification
        out = hnmp(m, H)
        criterion = nn.CrossEntropyLoss()
        # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        ce_loss = criterion(out, target)

        loss = dist_loss + args.ls * ce_loss

        opt.zero_grad()
        loss.backward()

        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))

        losses_per_epoch.append(loss.data.cpu().numpy())

        if np.isnan(losses_per_epoch[-1]):
            break_out_flag = True
            break

    if break_out_flag:
        print("** Failed training (NaN Loss)... **")
        break

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

    if epoch >= 0:
        with torch.no_grad():
            print("** Evaluating... **")
            if args.data == 'Inshop':
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif args.data != 'SOP':
                Recalls, NMIs = utils.evaluate_cos(model, dl_query, eval_nmi=False)
            else:
                Recalls, NMIs = utils.evaluate_cos_SOP(model, dl_query, eval_nmi=False)

        # Logging Evaluation Score
        if args.data == 'Inshop':
            for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
        elif args.data != 'SOP':
            for i in range(6):
                wandb.log({"R@{}".format(2 ** i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10 ** i): Recalls[i]}, step=epoch)

        # Best model
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch

            # Save model
            torch.save({'model_state_dict': model.state_dict()},
                       '{}/{}_{}_best.pth'.format(LOG_DIR, args.data, args.model))

            with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.data, args.model), 'w') as f:
                f.write('Parameters: {}\n\n'.format(vars(args)))
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.data == 'Inshop':
                    for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                        f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                elif args.data != 'SOP':
                    for i in range(6):
                        f.write("Best Recall@{}: {:.4f}\n".format(2 ** i, best_recall[i] * 100))
                else:
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(10 ** i, best_recall[i] * 100))

