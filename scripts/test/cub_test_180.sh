#!/usr/bin/env bash

# srun -N 1 --gres=gpu:1 -u --pty -p m40t4 \
#   python test_multiModel.py --net=BN-Inception \
#   --dim=512 --width=224 --batch_size=120 --data=car \
#   --data_root=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet \
#   --resume=log/Weight/car/checkpoints/car_ms_MultiRepeat_fulltest_origin/model_best.pth.tar \
#   --testAllAngle=True

srun -N 1 --gres=gpu:1 -u --pty -p m40t4 \
  python test_multiModel.py --net=BN-Inception \
  --dim=512 --width=224 --batch_size=120 --data=cub \
  --data_root=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/DataSet \
  --resume=/home/nfs/admin0/yaoguang/code/MetricLearning/Object-retrieval-others/log/Weight/cub/checkpoints/cub_ms_bs32_rotate180 \
  --feature_path=/home/nfs/admin0/yaoguang/data/train_data/MetricLearning/features \
  --testAllAngle=False --test_trans=center-crop-180 \