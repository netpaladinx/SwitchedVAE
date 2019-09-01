#!/usr/bin/env bash

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 1 --cuda_id 1 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > train_csvae_for_cars3d_1.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 2 --cuda_id 1 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > train_csvae_for_cars3d_2.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 3 --cuda_id 1 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 64 > train_csvae_for_cars3d_3.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 4 --cuda_id 1 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > train_csvae_for_cars3d_4.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 5 --cuda_id 1 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > train_csvae_for_cars3d_5.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset cars3d --exp_id 6 --cuda_id 1 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 64 > train_csvae_for_cars3d_6.log 2>&1 &
