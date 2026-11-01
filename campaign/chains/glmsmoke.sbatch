#!/usr/bin/env bash
#SBATCH --job-name=cf-glmsmoke
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; G=$S/glm; source $G/env.sh; cd /tmp
[ -f $S/v4/mixc_glm_s1/sft_train.jsonl ] || python $S/v4/v4_mix.py --contrast $S/data/curricula14b/random --gsm $S/v4/gsm_c --tokenizer $S/models/GLM-4.7-Flash --out $S/v4/mixc_glm_s1 --seed 1 2>&1 | tail -1
python -c "
from transformers import AutoTokenizer;import json
from probeforge.training_data import encode_row
t=AutoTokenizer.from_pretrained('$S/models/GLM-4.7-Flash')
for d in ['$S/data/curricula14b/random','$S/v4/mixc_glm_s1']:
    rows=[json.loads(l) for l in open(d+'/sft_train.jsonl')]; n=[len(encode_row(t,r,100000)['input_ids']) for r in rows]
    print(d.split('/')[-1],'rows',len(rows),'glm tokens',sum(n),'max',max(n))
"
cd ~/probeforge; rm -rf $S/runs/glmsmoke_*
srun --exact --ntasks=1 --cpus-per-task=40 --gres=gpu:2 --cpu-bind=none --output=$G/logs/smoke_random.out --error=$G/logs/smoke_random.err bash -c "source $G/env.sh; nvidia-smi -L; python -m contrastforge.cli train --config configs/vglm_smoke_random.json" < /dev/null &
srun --exact --ntasks=1 --cpus-per-task=40 --gres=gpu:2 --cpu-bind=none --output=$G/logs/smoke_v4c.out --error=$G/logs/smoke_v4c.err bash -c "source $G/env.sh; nvidia-smi -L; python -m contrastforge.cli train --config configs/vglm_smoke_v4c.json" < /dev/null &
wait; echo "end $(date)"; for n in random v4c; do echo "== $n"; tail -3 $G/logs/smoke_$n.out; tr '\r' '\n' < $G/logs/smoke_$n.err | grep -E "Error|error|it\]|s/it|trainable" | tail -6; cat $S/runs/glmsmoke_$n/run_manifest.json 2>/dev/null | grep -o '"trainable_parameters": [0-9]*'; done
