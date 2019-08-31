#!/usr/bin/env bash

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 1 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > train_csvae_for_dsprites_full_1.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 2 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > train_csvae_for_dsprites_full_2.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 3 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 64 > train_csvae_for_dsprites_full_3.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 4 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > train_csvae_for_dsprites_full_4.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 5 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > train_csvae_for_dsprites_full_5.log 2>&1 &

nohup python -u train_conv_switched_vae.py --dataset dsprites_full --exp_id 6 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 64 > train_csvae_for_dsprites_full_6.log 2>&1 &
