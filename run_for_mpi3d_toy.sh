#!/usr/bin/env bash

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 1 --channels 3 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 4 > run_for_mpi3d_toy_1.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 2 --channels 3 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 16 > run_for_mpi3d_toy_2.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 3 --channels 3 \
    --y_ce_beta 1 --y_phsic_beta 1 --y_mmd_beta 2 --z_beta 1 --z2_beta 64 > run_for_mpi3d_toy_3.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 4 --channels 3 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 4 > run_for_mpi3d_toy_4.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 5 --channels 3 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 16 > run_for_mpi3d_toy_5.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 6 --channels 3 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 8 --z_beta 4 --z2_beta 64 > run_for_mpi3d_toy_6.log 2>&1 &

nohup python -u run_fc_switched_vae.py --dataset mpi3d_toy --exp_id 7 --channels 3 \
    --y_ce_beta 4 --y_phsic_beta 4 --y_mmd_beta 6 --z_beta 8 --z2_beta 64 > run_for_mpi3d_toy_7.log 2>&1 &
