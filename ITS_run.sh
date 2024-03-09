#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 nohup python -u main.py --model_name MIMOUNet --mode train --train_data ITS-train --valid_data ITS-test --ex $2 \
  --data_dir /home/wuyabo/datasets/reside-indoor --num_epoch 500 \
  > $2.log 2>&1 &

#nohup \ --resume /home/wuyabo/pycode/SFNet-v1/dehaze_v2_sf_dataload_gap_densehaze/results/MIMOUNet/ITS-train/weight_ITS_test/model.pkl \
#  python -u main.py --model_name MIMOUNet --data Outdoor --mode train \
#  --data_dir /groups/public_cluster/home/wuyabo/datasets \
#  --batch_size 16 --num_epoch 30  --save_freq 1 --valid_freq 1 \
#  --resume ./results/MIMOUNet/Outdoor/weight_0719_17-45/model.pkl \
#  > output_out_16.log 2>&1 &
#--resume /home/wuyabo/pycode/SFNet-v1/dehaze_v2_sf_dataload_gap_densehaze/results/MIMOUNet/ITS-train/weight_ITSv5-22k1_1/model.pkl \
# nohup CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model_name MIMOUNet --data Indoor --mode train --data_dir /remote-home/wuyabo/datasets --batch_size 4 > output.log 2>&1 &
#CUDA_VISIBLE_DEVICES=5 nohup python -u main.py --model_name MIMOUNet --mode train --train_data ITS-train --valid_data ITS-test --ex 5 --data_dir /remote-home/wuyabo/datasets/reside-indoor --batch_size 4
