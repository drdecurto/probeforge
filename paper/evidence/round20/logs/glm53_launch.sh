L=/gpfs/scratch/bsc21/bsc021885/cf/logs
for i in $(seq 1 240); do grep -q "cubin done" $L/cubin.log && break; sleep 60; done
echo "cubins: $(tail -1 $L/cubin.log) $(date)"; bash $L/glm53_chain.sh
