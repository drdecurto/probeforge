S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE TRANSFORMERS_CACHE; export PIP_CACHE_DIR=/gpfs/scratch/bsc21/bsc021885/pip_cache; cd /tmp
BASEPY=$(readlink -f $S/envs/vllm32/bin/python)
# A) serving env: vLLM 0.29 + Aleph Alpha plugin (standalone venv)
[ -x $S/envs/kolvllm/bin/python ] || $BASEPY -m venv $S/envs/kolvllm
P=$S/envs/kolvllm/bin/python
for i in 1 2 3 4 5; do $P -m pip install -q --retries 10 --timeout 120 "vllm>=0.29,<0.30" "aleph-alpha-inference==1.0.0" && break; echo "serve install try $i failed"; sleep 20; done 2>&1 | tail -3
$P -c "import vllm,torch,transformers;print('kolvllm vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda,'tf',transformers.__version__)"
$P -c "import importlib.metadata as m;print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
# B) training env: layered on vllm32 (.pth) with transformers>=5.20 shadowing
rm -rf $S/envs/koltrain; $BASEPY -m venv --without-pip $S/envs/koltrain; SP=$S/envs/koltrain/lib/python3.12/site-packages
echo "$S/envs/vllm32/lib/python3.12/site-packages" > $SP/zz_vllm32.pth; echo "$HOME/probeforge/src" > $SP/zz_probeforge.pth
$S/envs/vllm32/bin/python -m pip install -q --retries 10 --no-deps --target $SP "transformers>=5.20,<5.21" "peft==0.18.1" accelerate 2>&1 | tail -2
$S/envs/koltrain/bin/python -c "import transformers,peft,torch;print('koltrain tf',transformers.__version__,transformers.__file__,'peft',peft.__version__,'torch',torch.__version__)"
echo KOLENVDONE
