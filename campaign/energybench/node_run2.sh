#!/usr/bin/env bash
# dynamic: each GPU claims missing (slot,lang) tasks; claims are per job (mkdir atomic), stale claims from older jobs ignored
S=/gpfs/scratch/bsc21/bsc021885/cf; B=$S/energybench; V=$S/envs/vllm32/bin
unset PYTHONPATH PYTHONUSERBASE; export HF_HOME=$S/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
SLOTS=${SLOTS:-$B/slots.txt}; OUTD=${OUTD:-$B/out}; EVALPY=${EVALPY:-$B/eval_patched.py}
LANGS="en es ca ar lb eu fr pt it de ru zh-cn zh-tw"; C=$B/claims/$SLURM_JOB_ID$(basename $OUTD); mkdir -p $C $B/out $B/logs
H=$(hostname -s)
gpu_loop() {
  k=$1; port=$((8100+k)); cur=""; SP=""
  while true; do
    task=""
    for spec in $(grep "^$cur|" $SLOTS) $(cat $SLOTS); do
      IFS='|' read slot base adapter <<< "$spec"
      for l in $LANGS; do [ -s $OUTD/$slot/$l.json ] && continue; mkdir $C/${slot}__$l 2>/dev/null && { task="$spec|$l"; break 2; }; done
    done
    [ -z "$task" ] && break
    IFS='|' read slot base adapter l <<< "$task"
    if [ "$slot" != "$cur" ]; then
      [ -n "$SP" ] && { kill $SP; wait $SP 2>/dev/null; sleep 5; }
      if [ -n "$adapter" ]; then EXTRA="--served-model-name $slot-base --enable-lora --max-lora-rank 16 --lora-modules $slot=$S/runs/$adapter"; else EXTRA="--served-model-name $slot"; fi
      CUDA_VISIBLE_DEVICES=$k $V/vllm serve $S/models/$base --port $port --max-model-len 4096 --gpu-memory-utilization 0.85 $EXTRA > $B/logs/serve_${slot}_${SLURM_JOB_ID}_$H$k.log 2>&1 &
      SP=$!
      for i in $(seq 1 60); do curl -s localhost:$port/v1/models | grep -q "\"$slot\"" && break; kill -0 $SP 2>/dev/null || break; sleep 10; done
      curl -s localhost:$port/v1/models | grep -q "\"$slot\"" || { echo "$slot: server failed on $H gpu $k"; tail -5 $B/logs/serve_${slot}_${SLURM_JOB_ID}_$H$k.log; rmdir $C/${slot}__$l; kill $SP 2>/dev/null; return; }
      cur=$slot; echo "$slot up on $H GPU $k $(date +%T)"
    fi
    mkdir -p $OUTD/$slot; W=$B/run_$(basename $OUTD)/$slot/$l; rm -rf $W; mkdir -p $W; cp $B/langs/$l/problems.json $W/
    ( cd $W && CUDA_VISIBLE_DEVICES=$k API_URL=http://localhost:$port/v1/completions OUT_JSON=$OUTD/$slot/$l.json.tmp \
        CHAT_TOKENIZER=$S/models/$base $V/python $EVALPY $slot $k > $B/logs/eval_$(basename $OUTD)_${slot}_$l.log 2>&1 ) && mv $OUTD/$slot/$l.json.tmp $OUTD/$slot/$l.json
    echo "$slot $l done $(date +%T) overall=$(python3 -c "import json;print(round(json.load(open('$OUTD/$slot/$l.json'))['analysis']['overall_average'],3))" 2>/dev/null)"
  done
  [ -n "$SP" ] && kill $SP; echo "gpu $H/$k idle $(date +%T)"
}
for k in 0 1 2 3; do gpu_loop $k & sleep 2; done; wait
