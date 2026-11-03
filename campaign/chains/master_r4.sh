S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; V=$S/v4; B=$S/energybench; SD=$S/stdbench; E=$S/evalr4; D=$S/data/contrast13m_200
source ~/probeforge/slurm/environment.sh; cd ~/probeforge
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable "$@" 2>/dev/null) && break; sleep 60; done; echo "$J"; }
echo "START $(date)"
trained() { while IFS='|' read cfg run; do [ -f $S/runs/$run/metrics.json ] || return 1; done < $V/train_list4.txt; }
for c in 1 2 3 4; do trained && break; J=$(submit $L/sft_r4.sbatch); echo "== sft r4 round$c $J $(date)"; waitjob $J; while IFS='|' read cfg run; do echo "  $run $(ls $S/runs/$run 2>/dev/null | grep checkpoint | sort -t- -k2 -n | tail -1) $([ -f $S/runs/$run/metrics.json ] && echo DONE)"; done < $V/train_list4.txt; done
evald() { while IFS='|' read name x; do [ -f $E/test_$name.jsonl.manifest.json ] || return 1; done < $V/eval_list4.txt; }
for c in 1 2 3; do evald && break; J=$(submit $L/eval_r4.sbatch); echo "== eval r4 $J"; waitjob $J; done
while IFS='|' read name x; do contrastforge evaluate --inputs $D/test_eval12.jsonl --rollouts $E/test_$name.jsonl --out $E/report12_$name.json < /dev/null >/dev/null
  python -c "import json;b=json.load(open('$E/report12_$name.json'))['budgets']['512'];print('RES $name pass@1=%.3f worst=%.3f boundary=%.2f cf=%.2f base_view=%.2f tok=%.0f'%(b['pass_at_1'],b['worst_language'],b['per_view']['boundary'],b['per_view']['counterfactual'],b['per_view']['base'],b['mean_completion_tokens']))"; done < $V/eval_list4.txt
emiss() { n=0; while IFS='|' read s b a; do for l in en es ca ar lb eu fr pt it de ru zh-cn zh-tw; do for o in out out_chat; do [ -s $B/$o/$s/$l.json ] || n=$((n+1)); done; done; done < $B/slots_v4c4.txt; echo $n; }
for c in 1 2 3 4; do m=$(emiss); echo "energy r4 missing $m $(date)"; [ $m = 0 ] && break; J=$(submit $L/energy_r4.sbatch); echo "== energy r4 $J"; waitjob $J; done
for c in 1 2; do n=0; while IFS='|' read name x; do [ -f $S/models/merged/$name/config.json ] || n=$((n+1)); done < $V/eval_list4.txt; [ $n = 0 ] && break; J=$(submit $L/stdmerge_r4.sbatch); echo "== stdmerge r4 $J"; waitjob $J; grep merge $L/stdmerger4-$J.out; done
for c in 1 2 3; do n=0; while IFS='|' read name x; do for p in mc mgsm; do [ -f $SD/out/$name/$p/DONE ] || n=$((n+1)); done; done < $V/eval_list4.txt; [ $n = 0 ] && break; J=$(submit $L/std_r4.sbatch); echo "== std r4 $J"; waitjob $J; done
python3 $SD/aggregate.py | grep -E "model|phi-4|aya"
echo "MASTER_R4 DONE $(date)"
