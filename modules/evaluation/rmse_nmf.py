import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_FILE = "data/ratings.csv"
TEST_FILE = "data/ratings.csv"
MODEL_PATH = "models_trained/model_NMF_rmse.pkl"
PRED_FILE = "results/preds_nmf_rmse.csv"
PAIRS_FILE = "results/_pairs_for_rmse.csv"
RESULTS_PKL = "results/nmf_rmse_runs.pkl"
N_RUNS = 10
BOOTSTRAP_BASE_SEED = 123


def run_main(arguments: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), *arguments]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_info() -> dict[str, object]:
    def read_git(*args: str) -> str:
        cmd = ["git", *args]
        output = subprocess.check_output(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return output.strip()

    try:
        commit = read_git("rev-parse", "HEAD")
        short_commit = read_git("rev-parse", "--short", "HEAD")
        branch = read_git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(read_git("status", "--porcelain"))
        return {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch,
            "is_dirty": dirty,
        }
    except Exception:
        return {
            "commit": "unknown",
            "short_commit": "unknown",
            "branch": "unknown",
            "is_dirty": None,
        }


def main() -> None:
    pairs_file = PROJECT_ROOT / PAIRS_FILE

    model_base = PROJECT_ROOT / MODEL_PATH
    pred_base = PROJECT_ROOT / PRED_FILE
    results_pkl = PROJECT_ROOT / RESULTS_PKL

    model_base.parent.mkdir(parents=True, exist_ok=True)
    pred_base.parent.mkdir(parents=True, exist_ok=True)
    pairs_file.parent.mkdir(parents=True, exist_ok=True)
    results_pkl.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROJECT_ROOT / TRAIN_FILE)
    test_file = PROJECT_ROOT / TEST_FILE
    test_df = pd.read_csv(test_file)
    required_cols = {"userId", "movieId", "rating"}
    missing = required_cols - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in TEST_FILE: {sorted(missing)}"
        )

    test_df[["userId", "movieId"]].to_csv(pairs_file, index=False)

    rmse_values = []
    run_results = []
    git_info = get_git_info()
    for run_idx in range(1, N_RUNS + 1):
        seed = BOOTSTRAP_BASE_SEED + run_idx
        run_train_path = (
            PROJECT_ROOT / f"results/_train_bootstrap_run_{run_idx}.csv"
        )
        run_model_path = model_base.with_name(
            f"{model_base.stem}_run_{run_idx}{model_base.suffix}"
        )
        run_pred_path = pred_base.with_name(
            f"{pred_base.stem}_run_{run_idx}{pred_base.suffix}"
        )
        run_started_at = utc_now_iso()

        train_boot = train_df.sample(
            n=len(train_df),
            replace=True,
            random_state=seed,
        )
        train_boot.to_csv(run_train_path, index=False)

        run_main(
            [
                "--mode",
                "train",
                "--train_file",
                str(run_train_path),
                "--model_path",
                str(run_model_path),
                "--alg",
                "NMF",
            ]
        )

        run_main(
            [
                "--mode",
                "predict",
                "--input_file",
                str(pairs_file),
                "--model_path",
                str(run_model_path),
                "--output_file",
                str(run_pred_path),
                "--alg",
                "NMF",
            ]
        )
        run_finished_at = utc_now_iso()

        with open(run_model_path, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        if hasattr(loaded_model, "get_model_metadata"):
            model_meta = loaded_model.get_model_metadata()
        else:
            model_meta = {
                "algorithm_name": loaded_model.__class__.__name__,
                "model_class": loaded_model.__class__.__name__,
                "model_params": {},
            }

        pred_df = pd.read_csv(run_pred_path)
        merged = test_df.merge(
            pred_df,
            on=["userId", "movieId"],
            suffixes=("_true", "_pred"),
        )
        rmse = float(
            np.sqrt(
                np.mean(
                    (merged["rating_true"] - merged["rating_pred"]) ** 2
                )
            )
        )
        rmse_values.append(rmse)
        run_results.append(
            {
                "run_idx": run_idx,
                "seed": seed,
                "rmse": rmse,
                "rows_used": len(merged),
                "train_file": str(run_train_path),
                "model_path": str(run_model_path),
                "pred_file": str(run_pred_path),
                "started_at_utc": run_started_at,
                "finished_at_utc": run_finished_at,
                "model_metadata": model_meta,
                "git": git_info,
            }
        )
        print(f"Run {run_idx:02d}: RMSE={rmse:.6f}")
        print(f"Run {run_idx:02d}: pred_file={run_pred_path}")

    summary = {
        "created_at_utc": utc_now_iso(),
        "n_runs": N_RUNS,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "train_file": str(PROJECT_ROOT / TRAIN_FILE),
        "test_file": str(PROJECT_ROOT / TEST_FILE),
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values)),
        "git": git_info,
        "runs": run_results,
    }
    with open(results_pkl, "wb") as output_file:
        pickle.dump(summary, output_file)

    print(f"Runs done: {N_RUNS}")
    print(f"RMSE mean: {float(np.mean(rmse_values)):.6f}")
    print(f"RMSE std: {float(np.std(rmse_values)):.6f}")
    print(f"Saved run metadata to: {results_pkl}")


if __name__ == "__main__":
    main()
