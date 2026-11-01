S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE TRANSFORMERS_CACHE; export PIP_CACHE_DIR=/gpfs/scratch/bsc21/bsc021885/pip_cache; cd /tmp
rm -rf $S/envs/glm; BASEPY=$(readlink -f $S/envs/vllm32/bin/python); echo base $BASEPY
$BASEPY -m venv --without-pip $S/envs/glm 2>/dev/null || $S/envs/vllm32/bin/python -m venv --without-pip $S/envs/glm
SP=$S/envs/glm/lib/python3.12/site-packages; echo "$S/envs/vllm32/lib/python3.12/site-packages" > $SP/_vllm32.pth
P=$S/envs/glm/bin/python
$P -c "import sys,transformers,torch;print('tf',transformers.__version__,transformers.__file__,'torch',torch.__version__)"
$P -c "import peft" 2>/dev/null && echo "peft already present" || $S/envs/vllm32/bin/python -m pip install -q --retries 10 --no-deps --target $SP "peft==0.18.1"
$P -c "import accelerate" 2>/dev/null && echo "accelerate present" || $S/envs/vllm32/bin/python -m pip install -q --retries 10 --no-deps --target $SP accelerate
echo "$HOME/probeforge/src" > $SP/_probeforge.pth
$P -c "import peft,accelerate,transformers,torch,contrastforge,probeforge;print('peft',peft.__version__,'acc',accelerate.__version__,'tf',transformers.__version__,'torch',torch.__version__)"
echo ENV3DONE
