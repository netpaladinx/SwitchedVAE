#!/usr/bin/env bash

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 1 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > run_for_cars3d_1.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 2 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > run_for_cars3d_2.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 3 --channels 1 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 64 > run_for_cars3d_3.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 4 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > run_for_cars3d_4.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 5 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > run_for_cars3d_5.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 6 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 64 > run_for_cars3d_6.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset cars3d --exp_id 7 --channels 1 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 6 --z_beta 8 --z2_beta 64 > run_for_cars3d_7.log 2>&1 &
