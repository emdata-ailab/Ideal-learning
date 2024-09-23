#!/usr/bin/env bash


DATA=product
CLASSNUM=11318 #98 for cars, 11318 for products
BatchSize=128
# DATA_ROOT=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet
DATA_ROOT=/workspace/ideal/IDEAL-Object-retrieval-others/DataSet
LOSS=opt
OPT_MARGINLB=0.4
OPT_MARGINUB=0.6
OPT_SCALE=1
CHECKPOINTS=log
R=.pth.tar
TRAIN_TRANS=rand-crop
TEST_TRANS=center-crop
LOG_NAME=product_opt_origin

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
srun -N 1 --gres=gpu:1 -u --pty -p 2080 -w em1  python3 train.py --net ${NET} \
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
--log_name $LOG_NAME \
--loss_balance=$LOSS_BALANCE \
--opt_marginlb=$OPT_MARGINLB \
--opt_marginub=$OPT_MARGINUB \
--opt_scale=$OPT_SCALE \
--freeze_BN \
# --loader_type $LOADER_TYPE \
