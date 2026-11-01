S=/gpfs/scratch/bsc21/bsc021885/cf; unset PYTHONPATH PYTHONUSERBASE; export PIP_CACHE_DIR=/gpfs/scratch/bsc21/bsc021885/pip_cache; cd /tmp
T=$S/envs/fi_cubin; mkdir -p $T
for i in 1 2 3 4 5 6; do $S/envs/vllm32/bin/python -m pip download --no-deps -d $S/envs/fi_wheels "flashinfer-cubin==0.7.0.post1" > $S/logs/cubinwheel_dl.log 2>&1 && break; echo "dl try $i failed: $(tail -1 $S/logs/cubinwheel_dl.log)"; sleep 10; done
ls -la $S/envs/fi_wheels
$S/envs/vllm32/bin/python -m pip install --no-deps --no-index --target $T $S/envs/fi_wheels/flashinfer_cubin-*.whl 2>&1 | tail -2
PYTHONPATH=$T $S/envs/vllm32/bin/python -c "import flashinfer_cubin,os;print('cubin pkg',flashinfer_cubin.__file__)"; du -sh $T; echo WHEELDONE
