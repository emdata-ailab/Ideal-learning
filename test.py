# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity
from utils.serialization import load_checkpoint
import torch 
import ast 
import pdb, os

parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--data', type=str, default='cub')
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                    help='Is gallery identical with query')
parser.add_argument('--net', type=str, default='VGG16-BN')
parser.add_argument('--resume', '-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument('--testAllAngle', type=str, default='False', help='whether test all angles')

parser.add_argument('--dim', '-d', type=int, default=512,
                    help='Dimension of Embedding Feather')
parser.add_argument('--width', type=int, default=224,
                    help='width of input image')

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                    help='if True extract feature from the last pool layer')

args = parser.parse_args()
# pdb.set_trace()
if args.testAllAngle.lower() == "ture":
    args.testAllAngle= True
else:
    args.testAllAngle= False
args_ = vars(args)
print('------------ Options -------------')
for k, v in sorted(args_.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

print('args.pool_feature is:', args.pool_feature)
if os.path.isdir(args.resume):
    resums = [os.path.join(args.resume, i) for i in sorted(os.listdir(args.resume))]
else:
    resums = [args.resume,]
recall_list = []
for resume in resums:
    checkpoint = load_checkpoint(resume)

    epoch = checkpoint['epoch']
    gallery_feature, gallery_labels, query_feature, query_labels = Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature, testAllAngle=True)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    if args.gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    
    recall_ks = Recall_at_ks(query_feat=query_feature, gallery_feat=gallery_feature, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)

    result = '  '.join(['%.4f' % k for k in recall_ks])
    print('**'*10,os.path.basename(resume),'**'*10)
    print('Epoch-%d' % epoch, result)
    recall_list.append(os.path.basename(resume)+" "+'Epoch-%d %s' % (epoch, result[0])+"\n")
with open('test.txt', 'w')as f:
    f.writelines(recall_list)
    