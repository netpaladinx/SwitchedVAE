#!/usr/bin/env bash

DATASET='dsprites_full'
CUDA_ID=0
CHANNELS=1
N_BRANCHES=3
N_SWITCHES=5
N_DIMS_SM=32
N_LATENT_Z2=10
CKPTS='step-110000.ckpt'

python -u eval_fc_switched_vae.py --dataset $DATASET --exp_id 1 --cuda_id $CUDA_ID \
    --channels $CHANNELS --n_branches $N_BRANCHES --n_switches $N_SWITCHES --n_dims_sm $N_DIMS_SM \
    --fc_operator_type 'I' --fc_switch_type 'I' --n_latent_z2 $N_LATENT_Z2 --ckpts $CKPTS \
    --y_ce_beta 1 --y_hsic_beta 1 --y_mmd_beta 2 --z_hsic_beta 1 --z_kl_beta 1 --z2_kl_beta 4 \
    > eval_fsvae_for_${DATASET}_1.log 2>&1 </dev/null

python -u eval_fc_switched_vae.py --dataset $DATASET --exp_id 2 --cuda_id $CUDA_ID \
    --channels $CHANNELS --n_branches $N_BRANCHES --n_switches $N_SWITCHES --n_dims_sm $N_DIMS_SM \
    --fc_operator_type 'I' --fc_switch_type 'I' --n_latent_z2 $N_LATENT_Z2 --ckpts $CKPTS \
    --y_ce_beta 1 --y_hsic_beta 1 --y_mmd_beta 2 --z_hsic_beta 1 --z_kl_beta 1 --z2_kl_beta 8 \
    > eval_fsvae_for_${DATASET}_2.log 2>&1 </dev/null
python -u eval_fc_switched_vae.py --dataset $DATASET --exp_id 3 --cuda_id $CUDA_ID \
    --channels $CHANNELS --n_branches $N_BRANCHES --n_switches $N_SWITCHES --n_dims_sm $N_DIMS_SM \
    --fc_operator_type 'II' --fc_switch_type 'II' --n_latent_z2 $N_LATENT_Z2 --ckpts $CKPTS \
    --y_ce_beta 1 --y_hsic_beta 1 --y_mmd_beta 2 --z_hsic_beta 1 --z_kl_beta 1 --z2_kl_beta 4 \
    > eval_fsvae_for_${DATASET}_3.log 2>&1 </dev/null

python -u eval_fc_switched_vae.py --dataset $DATASET --exp_id 4 --cuda_id $CUDA_ID \
    --channels $CHANNELS --n_branches $N_BRANCHES --n_switches $N_SWITCHES --n_dims_sm $N_DIMS_SM \
    --fc_operator_type 'II' --fc_switch_type 'II' --n_latent_z2 $N_LATENT_Z2 --ckpts $CKPTS \
    --y_ce_beta 1 --y_hsic_beta 1 --y_mmd_beta 2 --z_hsic_beta 1 --z_kl_beta 1 --z2_kl_beta 8 \
    > eval_fsvae_for_${DATASET}_4.log 2>&1 </dev/null
