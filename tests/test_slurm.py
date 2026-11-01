import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SlurmTests(unittest.TestCase):
    def test_contrast_single_process_and_matrix_arguments(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder)/"sbatch"
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": folder+os.pathsep+os.environ["PATH"], "CF_ACCOUNT":"test", "CF_QOS":"test"}
            for key in ("CF_GPUS", "CF_GRES", "CF_ARRAY", "CF_DEPENDENCY", "CF_PYTHON"):
                env.pop(key,None)
            command=["bash",str(repo/"slurm/submit_contrast.sh"),"train","--config","configs/contrast_sft_mn5.json"]
            args=json.loads(subprocess.check_output(command,env=env,text=True))
            self.assertIn("--gres=gpu:1",args);self.assertIn("--ntasks=1",args)
            env.update(CF_ARRAY="0-11%2",CF_DEPENDENCY="111:222")
            args=json.loads(subprocess.check_output(command,env=env,text=True))
            self.assertIn("--array=0-11%2",args);self.assertIn("--dependency=afterok:111:222",args)

    def test_submission_arguments_without_scheduler(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder)/"sbatch"
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": folder+os.pathsep+os.environ["PATH"], "PF_ACCOUNT": "test-account", "PF_QOS": "test-qos"}
            for key in ("PF_GPUS", "PF_PARTITION", "PF_DEPENDENCY"):
                env.pop(key, None)
            result = subprocess.run(["bash", str(repo/"slurm/submit.sh"), "sft", "configs/sft_mn5.json", "--max-steps", "30"], env=env, check=True, capture_output=True, text=True)
            args = json.loads(result.stdout)
            self.assertIn("--ntasks=1", args)
            self.assertIn("--gres=gpu:4", args)
            self.assertIn("--account=test-account", args)
            self.assertEqual(args[-4:], ["sft", "configs/sft_mn5.json", "--max-steps", "30"])


if __name__ == "__main__":
    unittest.main()
