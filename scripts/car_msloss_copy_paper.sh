#!/usr/bin/env bash

DATA=car
CLASSNUM=98 #98 for cars, 11318 for products
DATA_ROOT=/dataset/image_retrieval/
LOSS=Weight
CHECKPOINTS=/workspace/ideal/IDEAL-Object-retrieval-others/log
R=.pth.tar
LOG_NAME=car_msloss_rand_crop_2normal
TRAIN_TRANS=rand-crop
EXPERIMENT_TYPE=2normal

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
LR=1e-5   #######  
BatchSize=32 ####### 
NumInstances=4 ####### 
RATIO=0.16 
Width=224
OPTIM=ADAM ####### 
Momentum=0.9 
WeightDecay=0.0005 ####### 
BaseModelLrMul=1.0  ####### 
# ExperimentType=2normal ## add yang
#224 protocal: 227 for GoogleNet 224 for rest 

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/
if_exist_mkdir ${SAVE_DIR}


# if [ ! -n "$1" ] ;then
echo "Begin Training!"
CUDA_VISIBLE_DEVICES=0 python /workspace/ideal/IDEAL-Object-retrieval-others/train.py --net ${NET} \
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
--epoch 6000 \
--loss $LOSS \
--width ${Width} \
--save_dir ${SAVE_DIR} \
--save_step 20 \
--class_num ${CLASSNUM} \
--log_name=$LOG_NAME \
--ratio ${RATIO} \
--freeze_BN \
--train_trans $TRAIN_TRANS \
--experiment_type $EXPERIMENT_TYPE \
