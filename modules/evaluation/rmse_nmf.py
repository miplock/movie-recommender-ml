import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = "data/ratings.csv"
TEST_FILE = "data/ratings.csv"
MODEL_PATH = "models_trained/model_NMF_rmse.pkl"
PRED_FILE = "results/preds_nmf_rmse.csv"
PAIRS_FILE = "results/_pairs_for_rmse.csv"
N_RUNS = 10
BOOTSTRAP_BASE_SEED = 123


def run_main(arguments: list[str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), *arguments]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    pairs_file = PROJECT_ROOT / PAIRS_FILE

    model_base = PROJECT_ROOT / MODEL_PATH
    pred_base = PROJECT_ROOT / PRED_FILE

    model_base.parent.mkdir(parents=True, exist_ok=True)
    pred_base.parent.mkdir(parents=True, exist_ok=True)
    pairs_file.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"Run {run_idx:02d}: RMSE={rmse:.6f}")
        print(f"Run {run_idx:02d}: pred_file={run_pred_path}")

    print(f"Runs done: {N_RUNS}")
    print(f"RMSE mean: {float(np.mean(rmse_values)):.6f}")
    print(f"RMSE std: {float(np.std(rmse_values)):.6f}")


if __name__ == "__main__":
    main()
