S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs
waitjob() { [ -n "$1" ] || return; for i in $(seq 1 480); do squeue -h -j $1 2>/dev/null | grep -q . || break; sleep 30; done; sacct -j $1 -X -o jobid,state,elapsed,nodelist%40 -n; }
submit() { local J=""; for i in $(seq 1 720); do J=$(sbatch --parsable "$@" 2>/dev/null) && break; sleep 60; done; echo "$J"; }
dn() { for a in $1; do [ -f $S/runs/sftkol_$a/metrics.json ] || return 1; done; }
ck() { for a in $1; do echo "  $a: $(ls $S/runs/sftkol_$a 2>/dev/null | grep checkpoint | sort -t- -k2 -n | tail -1) $([ -f $S/runs/sftkol_$a/metrics.json ] && echo DONE)"; done; }
echo "START $(date)"
for c in 1 2 3 4 5 6 7 8; do dn "random_s1 random_s2 random_s3 v4c_s1 v4c_s2 v4c_s3" && break
  for trio in "random_s1 random_s2 random_s3" "v4c_s1 v4c_s2 v4c_s3"; do dn "$trio" && continue; J=$(submit --export=ALL,ARMS="$trio" $L/kolsft.sbatch); echo "== kolsft [$trio] round$c $J $(date)"; waitjob $J; ck "$trio"; done; done
echo "KOLSFT DONE $(date)"
