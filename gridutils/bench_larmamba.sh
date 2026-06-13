#!/bin/bash
# bench_larmamba.sh — worker: run the encoder inference benchmark on a GPU.
# Args: $1 polar_repo  $2 env_prefix  $3 larmamba_repo  $4 groups  $5 batch
set -euo pipefail
polar_repo=$1; env_prefix=$2; larmamba_repo=$3; groups=${4:-256,512,1024,2048}; batch=${5:-16}
export PATH="${env_prefix}/bin:${PATH}"
export PYTHONPATH="${larmamba_repo}:${polar_repo}:${PYTHONPATH:-}"
echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)  mode=${groups} batch=${batch}"
"${env_prefix}/bin/python" -c "import torch;print('GPU',torch.cuda.get_device_name())"
if [ "$groups" = "MIXER" ]; then
    "${env_prefix}/bin/python" -m larmamba.tests.bench_mixer --batch "$batch"
else
    "${env_prefix}/bin/python" -m larmamba.tests.bench_encoder --groups "$groups" --batch "$batch"
fi
echo "BENCH DONE"
