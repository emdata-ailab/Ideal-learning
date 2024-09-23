#!/usr/bin/env bash
DATA=car
DATA_ROOT=/home/zhiyuan.chen/Object-retrieval/DataSet
Gallery_eq_Query=True
LOSS=Self
CHECKPOINTS=ckps
R=.pth.tar

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
ALPHA=50
BETA=0.35
LR=1e-5
BatchSize=80
RATIO=0.16
Freeze_BN=False

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-S-${BETA}-M-${ALPHA}-freezebn-${Freeze_BN}
if_exist_mkdir ${SAVE_DIR}


# if [ ! -n "$1" ] ;then

echo "Begin Testing!"

Model_LIST=`seq  50 50 3000`
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=0 python3 test.py --net ${NET} \
    --data $DATA \
    --data_root ${DATA_ROOT} \
    --batch_size 20 \
    -g_eq_q ${Gallery_eq_Query} \
    --width 227 \
    -r ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee result/$LOSS/$DATA/${NET}-DIM-$DIM-Batchsize-${BatchSize}-ratio-${RATIO}-lr-$LR${POOL_FEATURE:+'-pool_feature'}-S-${BETA}-M-${ALPHA}.txt

done

