#!/usr/bin/env bash

DATASET='dsprites_full'
CUDA_ID=0
CHANNELS=1

DEBUG=$1

if [ -z "$DEBUG" ]; then
    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 1 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > train_fsvae_for_${DATASET}_1.log 2>&1 &

    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 2 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > train_fsvae_for_${DATASET}_2.log 2>&1 &

    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 3 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 2 --y_phsic_beta 2 --y_mmd_beta 4 --z_beta 2 --z2_beta 4 > train_fsvae_for_${DATASET}_3.log 2>&1 &

    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 4 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 2 --y_phsic_beta 2 --y_mmd_beta 4 --z_beta 2 --z2_beta 16 > train_fsvae_for_${DATASET}_4.log 2>&1 &

    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 5 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > train_fsvae_for_${DATASET}_5.log 2>&1 &

    nohup python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 6 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > train_fsvae_for_${DATASET}_6.log 2>&1 &
else
    python -u train_fc_switched_vae.py --dataset $DATASET --exp_id 1 --cuda_id $CUDA_ID --channels $CHANNELS \
        --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4
fi
