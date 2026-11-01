import os
import sys
import requests
import json
import re
import time
import threading
import math
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import seaborn as sns
from tqdm import tqdm

plt.switch_backend('agg')

# NVML (GPU power sampling)
try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    print("⚠️ pynvml not available. Energy metrics will not be computed.")

# ✅ Get model name from command-line argument
if len(sys.argv) < 2:
    print("Usage: python llm_custom_vllm_already_loaded_energy.py <model> [gpu_index]")
    sys.exit(1)

MODEL = sys.argv[1]
GPU_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SAVE_DIR = os.path.join("results", MODEL)
os.makedirs(SAVE_DIR, exist_ok=True)


class GPUEnergySampler:
    """Sample GPU power via NVML at a fixed rate and integrate to watt-hours."""
    def __init__(self, interval_sec: float = 0.1, gpu_index: int = 0):
        self.interval = interval_sec
        self.gpu_index = gpu_index
        self._thread = None
        self._stop = threading.Event()
        self.samples_watts = []
        self.start_time = None
        self.end_time = None
        self._enabled = NVML_AVAILABLE

    def __enter__(self):
        if not self._enabled:
            return self
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.start_time = time.time()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"⚠️ Failed to initialize NVML: {e}")
            self._enabled = False
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
                self.samples_watts.append(p_mw / 1000.0)  # -> watts
            except Exception:
                # If sampling fails, record NaN to keep cadence
                self.samples_watts.append(float("nan"))
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.end_time = time.time()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False

    def total_wh(self) -> float:
        """Calculate total watt-hours consumed."""
        if not self._enabled or not self.samples_watts:
            return float("nan")
        # Integrate watts over time using rectangle rule
        duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else (len(self.samples_watts) * self.interval)
        avg_watts = float(np.nanmean(self.samples_watts))
        watt_hours = avg_watts * (duration / 3600.0)
        return watt_hours

    def stats(self) -> dict:
        """Get statistics about power consumption."""
        if not self._enabled or not self.samples_watts:
            return {"enabled": False}
        arr = np.array(self.samples_watts, dtype=float)
        valid_samples = np.isfinite(arr)
        return {
            "enabled": True,
            "samples": int(valid_samples.sum()),
            "avg_watts": float(np.nanmean(arr)),
            "max_watts": float(np.nanmax(arr)),
            "min_watts": float(np.nanmin(arr)),
        }


class ImprovedReasoningEvaluator:
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str) -> List[float]:
        return self.sentence_model.encode([text])[0].tolist()

    def evaluate_reasoning(self, problem: Dict[str, Any], runs: int = 3) -> Dict[str, Any]:
        scores = []
        solutions = []
        for _ in range(runs):
            try:
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "prompt": f"Solve this problem step by step: {problem['problem']}",
                        "max_tokens": 300,
                        "temperature": 0.2
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                model_solution = result['choices'][0]['text']
                solutions.append(model_solution)

                score = self.calculate_score(model_solution, problem['solution'])
                scores.append(score)
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                return None

        step_scores = self.evaluate_steps(solutions[0], problem['solution_steps'])
        
        return {
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "solutions": solutions,
            "step_scores": step_scores,
            "average_step_score": np.mean(step_scores) if step_scores else None
        }

    def calculate_score(self, model_solution: str, correct_solution: str) -> float:
        model_embedding = self.generate_embedding(model_solution)
        solution_embedding = self.generate_embedding(correct_solution)
        similarity = cosine_similarity([model_embedding], [solution_embedding])[0][0]

        model_numbers = re.findall(r'\d+\.?\d*', model_solution)
        solution_numbers = re.findall(r'\d+\.?\d*', correct_solution)
        numeric_correctness = len(set(model_numbers) & set(solution_numbers)) / len(set(solution_numbers)) if solution_numbers else 1

        return (similarity + numeric_correctness) / 2

    def evaluate_steps(self, model_solution: str, solution_steps: List[str]) -> List[float]:
        model_steps = re.split(r'Step \d+:|^\d+\.\s', model_solution)
        model_steps = [step.strip() for step in model_steps if step.strip()]
        return [self.calculate_score(model_step, solution_step) for model_step, solution_step in zip(model_steps, solution_steps)]


def generate_problems():
    """Load problems from problems.json file."""
    with open("problems.json", "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_results(evaluations: List[Dict[str, Any]], problems: List[Dict[str, Any]], energy_data=None) -> Dict[str, Any]:
    analysis = {
        "overall_average": np.mean([eval["average_score"] for eval in evaluations if eval is not None]),
        "category_averages": defaultdict(list),
        "difficulty_averages": defaultdict(list),
        "step_accuracy": [],
        "consistency": []
    }
    
    for eval, problem in zip(evaluations, problems):
        if eval is not None:
            analysis["category_averages"][problem["category"]].append(eval["average_score"])
            analysis["difficulty_averages"][problem["difficulty"]].append(eval["average_score"])
            if eval["average_step_score"] is not None:
                analysis["step_accuracy"].append(eval["average_step_score"])
            analysis["consistency"].append(eval["score_std"])
    
    for category, scores in analysis["category_averages"].items():
        analysis["category_averages"][category] = np.mean(scores)
    for difficulty, scores in analysis["difficulty_averages"].items():
        analysis["difficulty_averages"][difficulty] = np.mean(scores)
    
    analysis["step_accuracy"] = np.mean(analysis["step_accuracy"]) if analysis["step_accuracy"] else None
    analysis["consistency"] = np.mean(analysis["consistency"]) if analysis["consistency"] else None
    
    # Add energy metrics if available
    if energy_data:
        analysis["energy"] = energy_data
    
    return analysis


def plot_results(analysis: Dict[str, Any]):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(analysis["category_averages"].keys()), y=list(analysis["category_averages"].values()))
    plt.title(f"Average Reasoning Scores by Category ({MODEL})")
    plt.xlabel("Category")
    plt.ylabel("Average Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "category_scores.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(analysis["difficulty_averages"].keys()), y=list(analysis["difficulty_averages"].values()))
    plt.title(f"Average Reasoning Scores by Difficulty ({MODEL})")
    plt.xlabel("Difficulty")
    plt.ylabel("Average Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "difficulty_scores.png"))
    plt.close()

    # Plot energy metrics if available
    if "energy" in analysis and analysis["energy"].get("enabled"):
        energy = analysis["energy"]
        
        # Energy efficiency plot
        plt.figure(figsize=(10, 6))
        metrics = {
            "Avg Power (W)": energy.get("gpu_avg_watts", 0),
            "Max Power (W)": energy.get("gpu_max_watts", 0),
            "Wh per 1K queries": energy.get("wh_per_1000_queries", 0)
        }
        plt.bar(metrics.keys(), metrics.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.title(f"Energy Metrics - {MODEL}")
        plt.ylabel("Value")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "energy_metrics.png"))
        plt.close()


def main():
    evaluator = ImprovedReasoningEvaluator(MODEL, os.environ.get("API_URL","http://localhost:8000/v1/completions"))
    problems = generate_problems()
    
    # Warmup (stabilize kernels before energy measurement)
    print("🔥 Running warmup query...")
    if problems:
        _ = evaluator.evaluate_reasoning(problems[0], runs=1)
    
    time.sleep(2)  # Allow GPU to stabilize

    num_prompts = len(problems)
    evaluations = []

    print(f"\n⚡ Starting evaluation with energy monitoring ({num_prompts} problems)...\n")

    with GPUEnergySampler(interval_sec=0.1, gpu_index=GPU_INDEX) as meter:
        wall_start = time.time()
        
        for problem in tqdm(problems, desc=f"Evaluating problems with {MODEL}"):
            evaluation = evaluator.evaluate_reasoning(problem)
            if evaluation:
                evaluations.append(evaluation)
                print(f"Problem: {problem['problem']}")
                print(f"Average Score: {evaluation['average_score']:.4f}")
                print(f"Score Std Dev: {evaluation['score_std']:.4f}")
                print(f"Step Scores: {[f'{score:.4f}' for score in evaluation['step_scores']]}\n")
            else:
                print(f"Failed to evaluate problem: {problem['problem']}\n")
        
        wall_end = time.time()

    total_wall_s = wall_end - wall_start
    total_wh = meter.total_wh()
    meter_stats = meter.stats()

    # Calculate AI Energy Score metrics
    if num_prompts > 0 and math.isfinite(total_wh):
        wh_per_1000 = float(total_wh / num_prompts * 1000.0)
    else:
        wh_per_1000 = float("nan")

    energy_data = {
        "enabled": meter_stats.get("enabled", False),
        "gpu_total_wh": total_wh,
        "gpu_avg_watts": meter_stats.get("avg_watts", float("nan")),
        "gpu_max_watts": meter_stats.get("max_watts", float("nan")),
        "gpu_min_watts": meter_stats.get("min_watts", float("nan")),
        "wall_time_sec": total_wall_s,
        "num_prompts": num_prompts,
        "wh_per_1000_queries": wh_per_1000,
        "samples_collected": meter_stats.get("samples", 0),
        "gpu_index": GPU_INDEX,
        "hardware_note": "Measured via NVML. Batch size = 1 per API call (3 runs per problem)."
    }

    print(f"\n📊 Energy Metrics:")
    print(f"   Total Energy: {total_wh:.4f} Wh")
    print(f"   Average Power: {meter_stats.get('avg_watts', 0):.2f} W")
    print(f"   Max Power: {meter_stats.get('max_watts', 0):.2f} W")
    print(f"   Wh per 1000 queries: {wh_per_1000:.4f}")
    print(f"   Wall time: {total_wall_s:.2f} seconds")

    analysis = analyze_results(evaluations, problems, energy_data)
    print(json.dumps(analysis, indent=2))
    if os.environ.get("OUT_JSON"):
        with open(os.environ["OUT_JSON"], "w") as _f:
            json.dump({"analysis": analysis, "problems": problems, "solutions": [e["solutions"] for e in evaluations],
                       "average_scores": [float(e["average_score"]) for e in evaluations]}, _f, ensure_ascii=False, default=float)
    plot_results(analysis)

    output_path = os.path.join(SAVE_DIR, "improved_reasoning_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "problems": problems,
            "evaluations": evaluations,
            "analysis": analysis
        }, f, indent=2)

    # Save energy metrics separately
    energy_output_path = os.path.join(SAVE_DIR, "energy_metrics.json")
    with open(energy_output_path, "w") as f:
        json.dump(energy_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"✅ Energy metrics saved to: {energy_output_path}")


if __name__ == "__main__":
    main()
