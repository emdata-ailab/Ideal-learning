#!/usr/bin/env bash

DATA=car
CLASSNUM=11318 #98 for cars, 11318 for products
DATA_ROOT=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet
LOSS=Self
freeze_BN=True
CHECKPOINTS=log
R=.pth.tar
TRAIN_TRANS=rand-crop-180
TEST_TRANS=center-crop-180
LOG_NAME=car_Trai180

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
ALPHA=30.0
BETA=0.35
BatchSize=75  # 16
NumInstances=5  # 4
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
python train.py \
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
--freeze_BN \
--train_trans $TRAIN_TRANS \
--test_trans $TEST_TRANS \
--log_name $LOG_NAME
# | tee self.kernel.datatorch.uniquetargets_l2_normself.kernel.datatorch.uniquetargets
# | tee log/$LOSS/$DATA/${NET}-DIM-$DIM-Batchsize-${BatchSize}-${OPTIM}-ratio-${RATIO}-lr-$LR-m-${BETA}-t-${ALPHA}-freezebn-${freeze_BN}-NumInstances-${NumInstances}-fixedlr

#--freeze_BN 
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


#1ssh 1: bs400_num25 2: bs75_num5  3: ADAM+fixed-lr1e-4_alpha30 4:  ADAM+fixed-lr1e-4_alpha50  5: alpha60  6: num5  
#7: bs75num5, 8:bs75num25
# srun -N 1 -w em2 --gres=gpu:1 --pty sh product.sh 
# srun -N 1 -w em2 --gres=mps:400 --pty sh 
# srun -N 1 --gres=gpu:1 --pty sh
# sh product.sh >/dev/null 2>&1 & 
