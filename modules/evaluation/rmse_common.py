import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RMSEEvalConfig:
    """Configuration container for bootstrap RMSE evaluation workflow.

    EN:
    Stores algorithm name, file paths, and run parameters used by the
    shared evaluation routine.

    Attributes:
        alg (str): Algorithm identifier passed to CLI, e.g. NMF or SVD1.
        train_file (str): Path to source ratings used for bootstrapping.
        test_file (str): Path to ratings used as RMSE reference.
        model_path (str): Base path for trained model files per run.
        pred_file (str): Base path for prediction files per run.
        pairs_file (str): Path for temporary user-movie pairs CSV.
        results_pkl (str): Output path for aggregated run metadata.
        n_runs (int): Number of bootstrap repetitions.
        bootstrap_base_seed (int): Seed base incremented per run.

    PL:
    Kontener konfiguracji dla procedury ewaluacji RMSE z bootstrapem.

    Przechowuje nazwę algorytmu, ścieżki plików i parametry uruchomień
    używane przez wspólną funkcję ewaluacji.

    Atrybuty:
        alg (str): Identyfikator algorytmu przekazywany do CLI, np. NMF.
        train_file (str): Ścieżka danych źródłowych do bootstrapu.
        test_file (str): Ścieżka danych referencyjnych dla RMSE.
        model_path (str): Bazowa ścieżka modeli trenowanych per run.
        pred_file (str): Bazowa ścieżka plików predykcji per run.
        pairs_file (str): Ścieżka tymczasowego CSV z parami user-film.
        results_pkl (str): Ścieżka wyjściowa agregatu metadanych runów.
        n_runs (int): Liczba powtórzeń bootstrapu.
        bootstrap_base_seed (int): Bazowe ziarno zwiększane na run.
    """
    alg: str
    train_file: str
    test_file: str
    model_path: str
    pred_file: str
    pairs_file: str
    results_pkl: str
    n_runs: int = 10
    bootstrap_base_seed: int = 123


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format.

    EN:
    Builds an aware UTC datetime and converts it to ISO string.

    Returns:
        str: Timestamp like 2026-04-01T14:00:00+00:00.

    PL:
    Zwraca bieżący znacznik czasu UTC w formacie ISO-8601.

    Tworzy obiekt daty/czasu z informacją o strefie UTC i zamienia
    go na łańcuch znaków ISO.

    Zwraca:
        str: Znacznik czasu, np. 2026-04-01T14:00:00+00:00.
    """
    return datetime.now(timezone.utc).isoformat()


def get_git_info(project_root: Path) -> dict[str, object]:
    """Collect basic git repository metadata for experiment logging.

    EN:
    Reads commit hash, short hash, branch name, and dirty flag for the
    repository rooted at the provided path.

    Args:
        project_root (Path): Root directory of the git repository.

    Returns:
        dict[str, object]: Dictionary with commit and working-tree status.

    PL:
    Pobiera podstawowe metadane git do logowania eksperymentu.

    Odczytuje hash commita, skrócony hash, nazwę gałęzi i flagę zmian
    niezacommitowanych dla repozytorium pod podaną ścieżką.

    Argumenty:
        project_root (Path): Katalog główny repozytorium git.

    Zwraca:
        dict[str, object]: Słownik z informacją o commicie i stanie drzewa.
    """
    def read_git(*args: str) -> str:
        cmd = ["git", *args]
        output = subprocess.check_output(
            cmd,
            cwd=str(project_root),
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


def run_main(project_root: Path, arguments: list[str]) -> None:
    """Execute project CLI entry point with provided argument list.

    EN:
    Calls `main.py` using the current Python interpreter and raises on
    non-zero exit code.

    Args:
        project_root (Path): Project root containing `main.py`.
        arguments (list[str]): CLI arguments passed after script name.

    Returns:
        None: Executes subprocess for its side effects.

    Raises:
        subprocess.CalledProcessError: If called command fails.

    PL:
    Uruchamia główny skrypt CLI projektu z przekazaną listą argumentów.

    Wywołuje `main.py` przez bieżący interpreter Pythona i zgłasza błąd
    przy niezerowym kodzie zakończenia.

    Argumenty:
        project_root (Path): Katalog projektu zawierający `main.py`.
        arguments (list[str]): Argumenty CLI po nazwie skryptu.

    Zwraca:
        None: Uruchamia proces potomny jako efekt uboczny.

    Wyjątki:
        subprocess.CalledProcessError: Gdy komenda zakończy się błędem.
    """
    cmd = [sys.executable, str(project_root / "main.py"), *arguments]
    subprocess.run(cmd, check=True, cwd=str(project_root))


def evaluate_bootstrap_rmse(
    project_root: Path,
    config: RMSEEvalConfig,
) -> None:
    """Run multi-seed bootstrap evaluation and save RMSE run metadata.

    EN:
    Builds bootstrap training samples, trains and predicts via project CLI,
    computes RMSE for each run, and stores summary with per-run details
    into a pickle file.

    Args:
        project_root (Path): Root path used to resolve relative files.
        config (RMSEEvalConfig): Algorithm and path configuration.

    Returns:
        None: Writes artifacts and summary to disk.

    Raises:
        ValueError: If test file lacks required columns.
        subprocess.CalledProcessError: If training/prediction subprocesses fail.

    PL:
    Uruchamia wielokrotną ewaluację bootstrap i zapisuje metadane RMSE.

    Tworzy bootstrapowe próbki treningowe, wykonuje trenowanie i predykcję
    przez CLI projektu, liczy RMSE dla każdego runu i zapisuje podsumowanie
    wraz ze szczegółami runów do pliku pickle.

    Argumenty:
        project_root (Path): Ścieżka główna do rozwiązywania plików względnych.
        config (RMSEEvalConfig): Konfiguracja algorytmu i ścieżek.

    Zwraca:
        None: Zapisuje artefakty i podsumowanie na dysk.

    Wyjątki:
        ValueError: Gdy plik testowy nie ma wymaganych kolumn.
        subprocess.CalledProcessError: Gdy trening/predykcja zakończy się błędem.
    """
    pairs_file = project_root / config.pairs_file
    model_base = project_root / config.model_path
    pred_base = project_root / config.pred_file
    results_pkl = project_root / config.results_pkl

    model_base.parent.mkdir(parents=True, exist_ok=True)
    pred_base.parent.mkdir(parents=True, exist_ok=True)
    pairs_file.parent.mkdir(parents=True, exist_ok=True)
    results_pkl.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(project_root / config.train_file)
    test_df = pd.read_csv(project_root / config.test_file)
    required_cols = {"userId", "movieId", "rating"}
    missing = required_cols - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in TEST_FILE: {sorted(missing)}"
        )

    test_df[["userId", "movieId"]].to_csv(pairs_file, index=False)

    rmse_values = []
    run_results = []
    git_info = get_git_info(project_root)

    for run_idx in range(1, config.n_runs + 1):
        seed = config.bootstrap_base_seed + run_idx
        run_train_path = (
            project_root / f"results/_train_bootstrap_run_{run_idx}.csv"
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
            project_root,
            [
                "--mode",
                "train",
                "--train_file",
                str(run_train_path),
                "--model_path",
                str(run_model_path),
                "--alg",
                config.alg,
            ],
        )

        run_main(
            project_root,
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
                config.alg,
            ],
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
        "n_runs": config.n_runs,
        "bootstrap_base_seed": config.bootstrap_base_seed,
        "train_file": str(project_root / config.train_file),
        "test_file": str(project_root / config.test_file),
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values)),
        "git": git_info,
        "runs": run_results,
    }
    with open(results_pkl, "wb") as output_file:
        pickle.dump(summary, output_file)

    print(f"Runs done: {config.n_runs}")
    print(f"RMSE mean: {float(np.mean(rmse_values)):.6f}")
    print(f"RMSE std: {float(np.std(rmse_values)):.6f}")
    print(f"Saved run metadata to: {results_pkl}")
