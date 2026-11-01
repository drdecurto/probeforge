S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE; export PIP_CACHE_DIR=/gpfs/scratch/bsc21/bsc021885/pip_cache
P=$S/envs/glm/bin/python
for i in 1 2 3 4 5; do $P -m pip install -q --retries 10 --timeout 60 "peft>=0.17" accelerate datasets && break; sleep 10; done 2>&1 | tail -3
$P -c "import peft,accelerate,transformers,torch;print('peft',peft.__version__,'acc',accelerate.__version__,'tf',transformers.__version__,'torch',torch.__version__)"
echo ENV2DONE
