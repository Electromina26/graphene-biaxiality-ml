#!/bin/bash
# Resumable MD campaign. usage: md_campaign.sh "<runs as th:T:lam ...>"
cd /home/user/graphene; export OMP_NUM_THREADS=1; P=/home/user/venv/bin/python; M=ft:/home/user/graphene/ft/ft_omat_r2_s2.model
for r in $1; do IFS=: read th T lam <<< "$r"
  tag=ft_omat_r2_s2_th${th}_T${T}_lam${lam}_s${SEED:-1}_v2; [ -f md/${tag}_final.xyz ] && continue
  $P md_strain.py $M $th $T $lam ${SEED:-1} > md/${tag}.log 2>&1 && echo "done $r $(grep FAIL md/${tag}.log)" || echo "FAILED $r"
done
