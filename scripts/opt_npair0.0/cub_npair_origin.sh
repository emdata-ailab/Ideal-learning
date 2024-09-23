#!/usr/bin/env bash

DATA=cub
CLASSNUM=100 #98 for cars, 11318 for products
# DATA_ROOT=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet
DATA_ROOT=/workspace/ideal/IDEAL-Object-retrieval-others/DataSet
LOSS=npair
NPAIRSCALE=1.0
CHECKPOINTS=log
LOG_NAME=cub_npair_origin

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

R=.pth.tar
NET=BN-Inception
DIM=512
ALPHA=30.0
BETA=0.35
BatchSize=128  # 16
NumInstances=4  # 4
RATIO=0.16
Width=224
BaseModelLrMul=0.1
OPTIM=SGD
LR=5e-2 #bn for 3e-4
Momentum=0.9
WeightDecay=0.0
#224 protocal: 227 for GoogleNet 224 for rest 

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/
if_exist_mkdir ${SAVE_DIR}

pwd
# if [ ! -n "$1" ] ;then
echo "Begin Training!"
srun -N 1 --gres=gpu:1 -u --pty -p nvidia11g -w em9 python train.py \
--momentum ${Momentum} \
--basemodel-lr-mul ${BaseModelLrMul} \
--weight-decay ${WeightDecay} \
--data $DATA \
--net ${NET} \
--data_root ${DATA_ROOT} \
--init random \
--lr $LR \
--dim $DIM \
--optim $OPTIM \
--alpha $ALPHA \
--beta $BETA \
--num_instances ${NumInstances} \
--batch_size ${BatchSize} \
--epoch 20000 \
--loss $LOSS \
--width ${Width} \
--save_dir ${SAVE_DIR} \
--save_step 10 \
--print_freq 20 \
--class_num ${CLASSNUM} \
--ratio ${RATIO} \
--log_name $LOG_NAME \
--freeze_BN \
--npairscale=$NPAIRSCALE \
