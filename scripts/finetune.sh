#!/bin/bash
# Fine-tune MACE-OMAT-0 (medium) on stress-controlled PBE data. usage: finetune.sh <seed> [epochs]
SEED=${1:-1}; EP=${2:-150}; TAG=${3:-ft_omat}
cd /home/user/graphene; export OMP_NUM_THREADS=1
mkdir -p ft && cd ft
/home/user/venv/bin/python -m mace.cli.run_train \
  --name="${TAG}_s${SEED}" --seed=$SEED \
  --foundation_model=/root/.cache/mace/maceomat0mediummodel \
  --multiheads_finetuning=False \
  --train_file=../train.xyz --valid_fraction=0.1 --test_file=../test.xyz \
  --energy_key=REF_energy --forces_key=REF_forces --stress_key=REF_stress \
  --E0s='{6: -1.26173507}' --model=MACE \
  --loss=universal --energy_weight=1.0 --forces_weight=10.0 --stress_weight=100.0 \
  --lr=0.001 --ema --ema_decay=0.99 --batch_size=4 --valid_batch_size=8 \
  --max_num_epochs=$EP --patience=40 --eval_interval=5 \
  --default_dtype=float64 --device=cpu --save_cpu --restart_latest \
  --compute_stress=True --error_table=PerAtomRMSEstressvirials
