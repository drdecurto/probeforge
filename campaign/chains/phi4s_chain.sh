S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; V=$S/v4; B=$S/energybench; SD=$S/stdbench; E=$S/evalphi4s; M=$S/models/phi-4
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable $1 2>/dev/null) && break; sleep 120; done; echo "$J"; }
echo "START $(date)"
while IFS='|' read name cfg seed; do python - "$name" "$cfg" "$seed" <<'PY'
import json,sys
name,cfg,seed=sys.argv[1:4]; c=json.load(open(cfg)); c.update(seed=int(seed), output_dir=f"/gpfs/scratch/bsc21/bsc021885/cf/runs/sftphi4_{name}")
json.dump(c,open(f"configs/vphi4_{name}.json","w"),indent=1)
PY
done < $V/phi4_seed_runs.txt
cat > $L/sftphi4s.sbatch <<'SB'
#!/usr/bin/env bash
#SBATCH --job-name=cf-sftphi4s
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
while IFS='|' read name cfg seed; do R=""; ls $S/runs/sftphi4_$name 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftphi4_$name/metrics.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftphi4s-$name.out --error=$L/sftphi4s-$name.err contrastforge train --config configs/vphi4_$name.json $R < /dev/null &
done < $S/v4/phi4_seed_runs.txt; wait
SB
cat > $L/evalphi4s.sbatch <<'SB'
#!/usr/bin/env bash
#SBATCH --job-name=cf-evalphi4s
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; E=$S/evalphi4s; T=$S/data/contrast13m_200/test_eval12.jsonl; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
while IFS='|' read name cfg seed; do [ -f $E/test_$name.jsonl.manifest.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$E/collect_$name.out --error=$E/collect_$name.err \
  contrastforge collect --inputs $T --out $E/test_$name.jsonl --model $S/models/phi-4 --adapter $S/runs/sftphi4_$name --budgets 512 --samples 1 < /dev/null &
done < $S/v4/phi4_seed_runs.txt; wait
SB
alltr() { while IFS='|' read n x; do [ -f $S/runs/sftphi4_$n/metrics.json ] || return 1; done < $V/phi4_seed_runs.txt; }
for c in 1 2 3; do alltr && break; J=$(submit $L/sftphi4s.sbatch); echo "== sftphi4s round$c $J $(date)"; waitjob $J; done
allev() { while IFS='|' read n x; do [ -f $E/test_$n.jsonl.manifest.json ] || return 1; done < $V/phi4_seed_runs.txt; }
for c in 1 2; do allev && break; J=$(submit $L/evalphi4s.sbatch); echo "== evalphi4s round$c $J $(date)"; waitjob $J; done
while IFS='|' read n x; do contrastforge evaluate --inputs $S/data/contrast13m_200/test_eval12.jsonl --rollouts $E/test_$n.jsonl --out $E/report12_$n.json < /dev/null >/dev/null
  python -c "import json;b=json.load(open('$E/report12_$n.json'))['budgets']['512'];print('PHI4 $n pass@1=%.3f worst=%.3f boundary=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['mean_completion_tokens']))"; done < $V/phi4_seed_runs.txt
# energy (chat only; raw gives empty completions for phi-4) for mined seeds
printf 'phi-4-mined_s1|phi-4|sftphi4_mined_s1\nphi-4-mined_s2|phi-4|sftphi4_mined_s2\n' > $B/slots_phi4s.txt
sed -e 's#slots_v4c3.txt#slots_phi4s.txt#g' -e 's#--job-name=energyv4c3#--job-name=energyphi4s#' -e 's#--nodes=2#--nodes=1#' -e '/eval_patched.py/d' -e 's#for i in 1; do#for i in 0; do#' $L/energy_v4c3.sbatch > $L/energy_phi4s.sbatch
miss() { n=0; while IFS='|' read s b a; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do [ -s $B/out_chat/$s/$l.json ] || n=$((n+1)); done; done < $B/slots_phi4s.txt; echo $n; }
for c in 1 2 3; do m=$(miss); echo "energy phi4s missing $m"; [ $m = 0 ] && break; J=$(submit $L/energy_phi4s.sbatch); echo "== energyphi4s $J"; waitjob $J; done
# stdbench for mined seeds (merged)
cat > $SD/merge_phi4s.sh <<'MS'
#!/usr/bin/env bash
S=/gpfs/scratch/bsc21/bsc021885/cf; SD=$S/stdbench; source ~/probeforge/slurm/environment.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1
for spec in "phi-4-mined_s1|sftphi4_mined_s1|0" "phi-4-mined_s2|sftphi4_mined_s2|1"; do IFS='|' read name run k <<< "$spec"; Mg=$S/models/merged/$name
  ( [ -f $Mg/config.json ] || CUDA_VISIBLE_DEVICES=$k python $SD/merge.py $S/models/phi-4 $S/runs/$run $Mg < /dev/null > $SD/logs/merge_$name.log 2>&1; echo "merge $name $([ -f $Mg/config.json ] && echo ok || echo FAIL)" ) &
done; wait
MS
chmod +x $SD/merge_phi4s.sh
grep -q "^phi-4-mined_s1|" $SD/models.txt || printf 'phi-4-mined_s1|merged/phi-4-mined_s1|\nphi-4-mined_s2|merged/phi-4-mined_s2|\n' >> $SD/models.txt
sed -e "s#$SD/merge_v4c3.sh#$SD/merge_phi4s.sh; $SD/node_std.sh#" -e 's#--job-name=stdmergev4c-3#--job-name=stdphi4s#' -e 's#--time=01:00:00#--time=02:00:00#' $L/stdmerge_v4c3.sbatch > $L/stdphi4s.sbatch
for c in 1 2; do n=0; for x in phi-4-mined_s1 phi-4-mined_s2; do for p in mc mgsm; do [ -f $SD/out/$x/$p/DONE ] || n=$((n+1)); done; done; [ $n = 0 ] && break
  J=$(submit $L/stdphi4s.sbatch); echo "== stdphi4s $J"; waitjob $J; grep -E "merge| ok | FAIL" $L/stdphi4s-$J.out | tail; done
python3 $SD/aggregate.py | grep -E "model|phi-4"
echo "CHAIN DONE $(date)"
