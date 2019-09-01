#!/usr/bin/env bash

DATASET='mpi3d_toy'
CHANNELS=3
CUDA_ID=2

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 1 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > train_csvae2_for_${DATASET}_01.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 2 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > train_csvae2_for_${DATASET}_02.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 3 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 64 > train_csvae2_for_${DATASET}_03.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 4 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > train_csvae2_for_${DATASET}_04.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 5 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > train_csvae2_for_${DATASET}_05.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 6 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 64 > train_csvae2_for_${DATASET}_06.log 2>&1 &

CUDA_ID=3

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 11 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 4 --z2_beta 4 > train_csvae2_for_${DATASET}_11.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 12 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 4 --z2_beta 16 > train_csvae2_for_${DATASET}_12.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 13 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 4 --z2_beta 64 > train_csvae2_for_${DATASET}_13.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 14 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > train_csvae2_for_${DATASET}_14.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 15 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 8 --z2_beta 16 > train_csvae2_for_${DATASET}_15.log 2>&1 &

nohup python -u train_conv_switched_vae_v2.py --dataset $DATASET --exp_id 16 --cuda_id $CUDA_ID --channels $CHANNELS \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 8 --z2_beta 64 > train_csvae2_for_${DATASET}_16.log 2>&1 &