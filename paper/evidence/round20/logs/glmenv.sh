S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE; export PIP_CACHE_DIR=/gpfs/scratch/bsc21/bsc021885/pip_cache
[ -x $S/envs/glm/bin/python ] || $S/envs/vllm32/bin/python -m venv --system-site-packages $S/envs/glm
P=$S/envs/glm/bin/python
$P -m pip install -q "peft>=0.17" accelerate datasets 2>&1 | tail -3
$P -m pip install -q --no-deps -e ~/probeforge 2>&1 | tail -3
$P -c "import peft,accelerate,transformers,torch;print('peft',peft.__version__,'acc',accelerate.__version__,'tf',transformers.__version__,'torch',torch.__version__)"
$P -c "import contrastforge,probeforge;print('cf ok')"
echo ENVDONE
