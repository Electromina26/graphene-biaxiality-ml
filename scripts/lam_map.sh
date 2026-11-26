#!/bin/bash
# Resumable lambda-theta phonon map. usage: lam_map.sh <model-file> "<thetas>" "<lams>"
M=ft:$1; THS=$2; LAMS=$3
cd /home/user/graphene; export OMP_NUM_THREADS=1; P=/home/user/venv/bin/python
E=0.12,0.13,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.20,0.205,0.21,0.215,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29
TAG=$(echo $M | sed 's/ft:/ft_/; s#/#_#g')
for th in $THS; do for l in $LAMS; do
  if [ "$l" = "0" ]; then f=phon/${TAG}_th${th}.json; else f=phon/${TAG}_th${th}_lam${l}.json; fi
  [ -f $f ] && continue
  $P phonons.py $M $th $E $l > /dev/null 2>&1 && echo "done th $th lam $l" || echo "FAILED th $th lam $l"
done; done
