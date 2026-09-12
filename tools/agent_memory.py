import json
import os

class AgentMemory:
    def __init__(self, storage_file: str = "experiment_memory.json"):
        self.storage_file = storage_file
        self.experiments = []
        self.reset()

    def reset(self):
        self.experiments = []
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)

    def log_experiment(self, config: dict, metrics: dict, top_features: dict) -> dict:
        exp_id = len(self.experiments) + 1
        entry = {
            "experiment_id": exp_id,
            "config": config,
            "metrics": metrics,
            "top_features": top_features
        }
        self.experiments.append(entry)
        self._save()
        return entry

    def get_best_experiment(self) -> dict:
        if not self.experiments:
            return {}

        is_classification = "f1_score" in self.experiments[0]["metrics"]
        key = "f1_score" if is_classification else "r2_score"

        best = max(self.experiments, key=lambda x: x["metrics"].get(key, -999))
        return {
            "best_experiment_id": best["experiment_id"],
            "model_name": best["config"]["model_name"],
            "best_metrics": best["metrics"],
            "total_experiments_conducted": len(self.experiments)
        }

    def _save(self):
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)