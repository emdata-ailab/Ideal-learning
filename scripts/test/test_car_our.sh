#!/usr/bin/env bash

# srun -N 1 --gres=gpu:1 -u --pty -p m40t4 \
#   python test_multiModel.py --net=BN-Inception \
#   --dim=512 --width=224 --batch_size=120 --data=car \
#   --data_root=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet \
#   --resume=log/Weight/car/checkpoints/car_ms_MultiRepeat_fulltest_origin/model_best.pth.tar \
#   --testAllAngle=True

# srun -N 1 --gres=gpu:1 -u --pty -p m40t4 \
#   python test_multiModel.py --net=BN-Inception \
#   --dim=512 --width=224 --batch_size=120 --data=car \
#   --data_root=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet \
#   --resume=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/log/Weight/car/checkpoints/car_ms_bs128_baseline_/model_best.pth.tar \
#   --feature_path=/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features/car_ms_bs128_baseline_ \
#   --testAllAngle=True --test_trans=center-crop \


CUDA_VISIBLE_DEVICES=0 python /workspace/ideal/IDEAL-Object-retrieval-others/test_multiModel.py --net=BN-Inception \
--dim=512 --width=224 --batch_size=120 --data=car \
--data_root=/dataset/image_retrieval/ \
--resume=/workspace/ideal/IDEAL-Object-retrieval-others/log/Weight/car/checkpoints/car_ms_bs32_baseline/model_best.pth.tar  \
--feature_path=/workspace/ideal/IDEAL-Object-retrieval-others/MetricLearning/features/car_ms_bs32_baseline \
--testAllAngle=True --test_trans=center-crop \
