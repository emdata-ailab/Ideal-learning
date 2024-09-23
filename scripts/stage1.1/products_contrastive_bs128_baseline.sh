#!/usr/bin/env bash


DATA=product
CLASSNUM=11318 #98 for cars, 11318 for products
BatchSize=128
DATA_ROOT=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet
LOSS=Contrastive
CHECKPOINTS=log
R=.pth.tar
LOG_NAME=procduct_Contrastive_bs128_baseline
LOADER_TYPE=multi_repeat

if_exist_mkdir ()
{
    dirname=$1
    if [ ! -d "$dirname" ]; then
    mkdir $dirname
    fi
}

if_exist_mkdir ${CHECKPOINTS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DATA}

if_exist_mkdir result
if_exist_mkdir result/${LOSS}
if_exist_mkdir result/${LOSS}/${DATA}

if_exist_mkdir log
if_exist_mkdir log/${LOSS}
if_exist_mkdir log/${LOSS}/${DATA}

NET=BN-Inception
DIM=512
ALPHA=40.0
BETA=2.0
LR=1e-6

NumInstances=4
RATIO=0.16
Width=224
OPTIM=ADAM
Momentum=0.9
WeightDecay=0.0 #0.0001
BaseModelLrMul=1.0
#224 protocal: 227 for GoogleNet 224 for rest 

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/
if_exist_mkdir ${SAVE_DIR}


# if [ ! -n "$1" ] ;then
echo "Begin Training!"
CUDA_VISIBLE_DEVICES=0 python3 train.py --net ${NET} \
--data $DATA \
--basemodel-lr-mul ${BaseModelLrMul} \
--optim $OPTIM \
--momentum ${Momentum} \
--weight-decay ${WeightDecay} \
--data_root ${DATA_ROOT} \
--init random \
--lr $LR \
--dim $DIM \
--alpha $ALPHA \
--beta $BETA \
--num_instances ${NumInstances} \
--batch_size ${BatchSize} \
--epoch 20000 \
--loss $LOSS \
--width ${Width} \
--save_dir ${SAVE_DIR} \
--save_step 2 \
--class_num ${CLASSNUM} \
--ratio ${RATIO} \
--loader_type $LOADER_TYPE \
--log_name $LOG_NAME \
#--freeze_BN \

# DATA=product
# CLASSNUM=11318 #98 for cars, 11318 for products
# DATA_ROOT=/home/nfs/admin0/chenzhiyuan/Object-retrieval-others/DataSet
# LOSS=Weight
# CHECKPOINTS=log
# R=.pth.tar

# if_exist_mkdir ()
# {
#     dirname=$1
#     if [ ! -d "$dirname" ]; then
#     mkdir $dirname
#     fi
# }

# if_exist_mkdir ${CHECKPOINTS}
# if_exist_mkdir ${CHECKPOINTS}/${LOSS}
# if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DATA}

# if_exist_mkdir result
# if_exist_mkdir result/${LOSS}
# if_exist_mkdir result/${LOSS}/${DATA}

# if_exist_mkdir log
# if_exist_mkdir log/${LOSS}
# if_exist_mkdir log/${LOSS}/${DATA}

# NET=BN-Inception
# DIM=512
# ALPHA=40.0
# BETA=2.0
# LR=1e-5
# BatchSize=80
# NumInstances=5
# RATIO=0.16
# Width=224
# OPTIM=ADAM
# Momentum=0.9
# WeightDecay=0.0005
# BaseModelLrMul=0.1
# #224 protocal: 227 for GoogleNet 224 for rest 

# SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/
# if_exist_mkdir ${SAVE_DIR}


# # if [ ! -n "$1" ] ;then
# echo "Begin Training!"
# CUDA_VISIBLE_DEVICES=0 python3 train.py --net ${NET} \
# --data $DATA \
# --basemodel-lr-mul ${BaseModelLrMul} \
# --optim $OPTIM \
# --momentum ${Momentum} \
# --weight-decay ${WeightDecay} \
# --data_root ${DATA_ROOT} \
# --init random \
# --lr $LR \
# --dim $DIM \
# --alpha $ALPHA \
# --beta $BETA \
# --num_instances ${NumInstances} \
# --batch_size ${BatchSize} \
# --epoch 20000 \
# --loss $LOSS \
# --width ${Width} \
# --save_dir ${SAVE_DIR} \
# --save_step 2 \
# --class_num ${CLASSNUM} \
# --ratio ${RATIO} \
# #--freeze_BN \
# | tee log/$LOSS/$DATA/${NET}-DIM-$DIM-Batchsize-${BatchSize}-ratio-${RATIO}-lr-$LR-S-${BETA}-M-${ALPHA}-freezebn-${freeze_BN}-NumInstances-${NumInstances}-SGD-autolr-layer-bnright-proxyunlearn-onlyposLSE.txt

#mvp bs 75, num25 [1, 1, 1] => 67.26
#mvp bs 64, num16 tolerance[1, 1, 1] => ckps/Mvp/car/resnet50-DIM-2048-lr1e-4-ratio-0.16-BatchSize-64-S-400.0-M-200.0-freezebn-False-NumInstances-16 => 0.7123  
#mvp bs 64, num16 tolerance[1, 5, 5] => mvp_64_16_155.log => 0.7318
#mvptriplet bs 64, num16 tolerance[1, 5, 5] =>  0.7257 (收敛太慢)

#mvp bs 64, num16 tolerance[1, 5, 5] same100,diff200 => mvp_64_16_155_same100_diff200-2019-06-17-15-49-22 => 0.7519
#same100,diff100 => mvp_64_16_155_same100_diff100-2019-06-17-16-03-25 => 0.7559
#same200,diff200 => mvp_64_16_155_same200_diff200-2019-06-17-16-02-43 => 0.7573
#same200,diff250 => mvp_64_16_155_same200_diff250-2019-06-17-16-00-16 => 0.7815
#same100,150 => mvp_64_16_155_same100_diff150-2019-06-17-21-17-44 => 0.7773
#same200,diff300 => mvp_64_16_155_same200_diff300-2019-06-17-21-20-16 => 0.7720
#same200,diff250 + fc2048 => mvp_64_16_155_same200_diff250_fc2048 => 0.7706
#same200,diff250 + fc2048 + fc_kaiminginit => mvp_64_16_155_same200_diff250_fc2048_kai => 0.7828
#same200,diff250 + fc2048 + fc_kaiminginit + bn_norm => ?
#same200,diff250 + layer_norm => mvp_64_16_155_same200_diff250_layernorm-2019-06-17-21-31-06 => 0.0455 (loss_pos 2000+）
#same200,diff250 + bn_norm => mvp_64_16_155_same200_diff250_bnnorm-2019-06-17-21-34-27 => 0.7309 (loss_pos 1000+）
#same200,diff250 + fc + bn_norm  => mvp_64_16_155_same200_diff250_bnnorm_fc-2019-06-17-21-37-54 => 0.7699
#same200,diff250 + fc2048 + bn_norm => mvp_64_16_155_same200_diff250_bnnorm_fc2048-2019-06-18-12-53-07 => 0.04
#same200,diff250 + fc + bn_norm + fc_kaiminginit + bninit =>mvp_64_16_155_same200_diff250_bnnorm_fc_fckaiinit_bninit-2019-06-18-12-33-15=>0.7594
#same200,diff250 + fc + bn_norm + fc_kaiminginit =>mvp_64_16_155_same200_diff250_bnnorm_fc_fckaiinit-2019-06-18-12-35-13=>0.7565
#same200,diff250 + fc2048 + bn_norm + fc_kaiminginit  => mvp_64_16_155_same200_diff250_bnnorm_fc2048_fckaiinit => 0.7267
#same200,diff250 + fc + relu + bn_norm + fc_kaiminginit + bninit => mvp_64_16_155_same200_diff250_bnnorm_fc_relu_fckaiinit_bninit-2019-06-18-12-39-14 => 0.7466
#same200,diff250 + fc2048 + relu + bn_norm + fc_kaiminginit + bninit => mvp_64_16_155_same200_diff250_bnnorm_fc2048_relu_fckaiinit_bninit-2019-06-18-12-44-18 => 0.04
# mvp_128_32_155_same200_diff250-2019-06-19-19-47-21 => 0.8140


#cub200_mvp_64_16_155_same200_diff250 => cub200_mvp_64_16_155_same200_diff250-2019-06-25-11-31-25 => ?
#cub200_mvp_128_32_155_same200_diff250=> cub200_mvp_128_32_155_same200_diff250-2019-06-25-11-36-50 => ?
#cub200_mvp_64_16_155_same200_diff300-2019-06-25-11-40-17 => ?


#Recall: Epoch-169 0.6140 0.7171  0.8113  0.8832  0.9338  0.9630
#Args:Namespace(alpha=250.0, batch_size=128, beta=300.0, data='cub', data_root='/home/zhiyuan.chen/Object-retrieval/DataSet', dim=512, epochs=20000, feature_norm=False, freeze_BN=False, init='random', k=16, log_name='cub200_mvp_128_32_122222_same200_diff250', loss='Mvp', loss_base=0.75, lr=1e-05, margin=0.5, momentum=0.9, nThreads=16, net='BN-Inception', num_instances=32, origin_width=256, orth_reg=0, print_freq=20, ratio=0.16, resume=None, save_dir='log/Mvp/cub/', save_step=10, weight_decay=0.0005, width=227)
