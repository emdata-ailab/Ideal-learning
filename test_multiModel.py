# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity
from utils.serialization import load_checkpoint

import numpy as np
import torch 
import ast 
import pdb, os


def main(args):
    print('args.pool_feature is:', args.pool_feature)

    if os.path.isdir(args.resume):
        resums = sorted([os.path.join(args.resume, i) for i in sorted(os.listdir(args.resume)) if i.startswith("Epoch")])
    else:
        resums = [args.resume,]
    # pdb.set_trace()
    feature_path = touch_dir(os.path.join(args.feature_path, os.path.basename(args.resume)))

    recall_list = []
    for resume in resums:
        checkpoint = load_checkpoint(resume)
        epoch = checkpoint['epoch']
        # if epoch % 20 != 0: continue
        feature_pa = os.path.join(feature_path, f'Epoch_{epoch}.npz')
        if not os.path.exists(feature_pa):
            _, _, query_feature, query_labels = Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature, testAllAngle=args.testAllAngle, test_trans=args.test_trans)
        else:
            np_data = np.load(feature_pa)
            query_feature = torch.from_numpy(np_data['feature']).cuda()
            query_labels = np_data['labels']
        
        recall_ks = Recall_at_ks(query_feat=query_feature, gallery_feat=query_feature, query_ids=query_labels, gallery_ids=query_labels, data=args.data)

        result = '  '.join(['%.4f' % k for k in recall_ks])
        print('**'*10,os.path.basename(resume),'**'*10)
        print('Epoch-%d' % epoch, result)
        recall_list.append('Epoch-%d %f' % (epoch, recall_ks[0])+"\n")
        if not os.path.exists(feature_pa):
            np.savez(feature_pa, feature=query_feature.cpu().numpy(), labels=torch.stack(query_labels).cpu().numpy())

    with open(os.path.join(feature_path, "recall_list.txt"), 'w')as f:
        f.writelines(recall_list)


def recall_cat_feature(feature_path_dict, args):
    npys_dict = {}
    length = 9999

    path_score = {}
    for key, path in feature_path_dict.items():
        for npy_name in [i for i in os.listdir(path) if i.endswith("npz")]:
            if npy_name not in path_score:
                path_score[npy_name] = 1
            else:
                path_score[npy_name] += 1

    path_list = [key for key, value in path_score.items() if value == 4]
    path_d = {int(pa[6:-4]):pa for pa in path_list}
    inds = sorted(path_d.keys())
    path_list = [path_d[ind] for ind in inds]

    # pdb.set_trace()
    def _recall(npzlist):
        if len(npzlist) > 1:
            feature_list, label_list = [], []
            for npz_file in npzlist:
                try:
                    npz_data = np.load(npz_file)
                except:
                    print("YYYY", npz_file, npzlist)
                    exit(0)
                feature_list.append(npz_data['feature'])
                label_list.append(npz_data['labels'])
            # pdb.set_trace()
            if len(label_list) > 1:
                assert (label_list[0] == label_list[1]).all()
            feature = torch.from_numpy(np.concatenate(feature_list, axis=1)).cuda()
            label = label_list[0]
        else:
            try:
                npz_data = np.load(npzlist[0])
            except:
                print("XXXX", npz_file)
                exit(0)
            feature = torch.from_numpy(npz_data['feature'])
            label = npz_data['labels']
        
        recall_ks = Recall_at_ks(query_feat=feature, gallery_feat=feature, query_ids=label, gallery_ids=label, data=args.data)
        return recall_ks[0]

    # pdb.set_trace()
    recall_list = ["epoch cat angle0 angle90 angle180 angle270\n"]
    for npy_name in path_list:
        # pdb.set_trace()
        recall,ps = {}, []
        for key, path in feature_path_dict.items():
            npy_path = os.path.join(path, npy_name)
            ps.append(npy_path)
            assert os.path.exists(npy_path)
            recall[key] = _recall([npy_path, ])
        
        recall1 = _recall(ps)
        strs = npy_name[:-4]+" %.04f  %.04f  %.04f  %.04f  %.04f\n"%(recall1, recall["0"], recall["90"],recall["180"],recall["270"])
        recall_list.append(strs)
        print(strs)

    # recall1 = _recall(best_list)
    # print('best cat:', recall1)
    # recall_list.append("Best"+" "+str(recall1)+"\n")
    # with open(f"/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/cat_{args.data}.txt", "w")as f:
    #     f.writelines(recall_list)
    with open(f"/workspace/ideal/IDEAL-Object-retrieval-others/MetricLearning/features/cat_{args.data}.txt", "w")as f:
        f.writelines(recall_list) # edit yangzl


def read_npz(path):
    npz_data = np.load(path)
    return npz_data['feature'], npz_data['labels']

def touch_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--feature_path', type=str, default='')
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')
    parser.add_argument('--net', type=str, default='VGG16-BN')
    parser.add_argument('--resume', '-r', type=str, default='model.pkl', metavar='PATH')
    parser.add_argument('--testAllAngle', type=str2bool, nargs='?', default=False, const=False, help='whether test all angles')

    parser.add_argument('--test_trans', type=str, default='center-crop', help='train dataset transform type')

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
    return args

if __name__ == "__main__":
    args = args_parser()
    args_ = vars(args)
    print('------------ Options -------------')
    for k, v in sorted(args_.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(args)

    # if args.data == 'car':
    #     feature_path_dict = {
    #         "0": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/car_ms_bs32_baseline", 
    #         "90": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/car_ms_bs32_rotate90", 
    #         "180": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/car_ms_bs32_rotate180", 
    #         "270": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/car_ms_bs32_rotate270", 
    #     }
    # elif args.data == 'cub':
    #     feature_path_dict = {
    #         "0": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/cub_ms_bs32_baseline", 
    #         "90": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/cub_ms_bs32_rotate90", 
    #         "180": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/cub_ms_bs32_rotate180", 
    #         "270": "/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/cub_ms_bs32_rotate270", 
    #     }
    # recall_cat_feature(feature_path_dict, args)
