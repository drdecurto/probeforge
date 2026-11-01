#!/usr/bin/env bash
#SBATCH --job-name=cf-gputest
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
source ~/probeforge/slurm/environment.sh
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch,time
print("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.get_device_name(0))
a=torch.randn(8192,8192,device="cuda",dtype=torch.bfloat16);torch.cuda.synchronize();t=time.time()
for _ in range(20): b=a@a
torch.cuda.synchronize();dt=time.time()-t;print(f"bf16 matmul TFLOPS ~ {20*2*8192**3/dt/1e12:.0f}")
PY
contrastforge doctor
