# Movie Recommender System (Project 1)
### Mikołaj (s324671)

Implementacja projektu 1 z MCaDR oparta o interfejs CLI w `main.py`.

Aktualnie zaimplementowane:
- trenowanie modelu (`--mode train`)
- predykcja ocen (`--mode predict`)
- algorytmy: `NMF`, `SVD1`
- skrypty ewaluacji RMSE (bootstrap):
  - `modules/evaluation/rmse_nmf.py`
  - `modules/evaluation/rmse_svd1.py`

Algorytmy wymagane w specyfikacji, ale jeszcze niezaimplementowane:
- `SVD2`
- `SGD`
- `BEST`

## Wymagania

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Struktura projektu

```text
project1_s324671/
├── data/
│   └── ratings.csv
├── modules/
│   ├── models/
│   │   ├── base_model.py
│   │   ├── nmf_model.py
│   │   └── svd1_model.py
│   ├── evaluation/
│   │   ├── rmse_common.py
│   │   ├── rmse_nmf.py
│   │   └── rmse_svd1.py
│   ├── train.py
│   └── predict.py
├── models_trained/
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Format danych

Plik treningowy (`ratings.csv`) musi zawierać kolumny:
- `userId`
- `movieId`
- `rating`

Plik wejściowy do predykcji musi zawierać:
- `userId`
- `movieId`

Plik wyjściowy predykcji (`preds.csv`) zawiera:
- `userId`
- `movieId`
- `rating`

Predykcje są przycinane do przedziału `[0.5, 5.0]` i zaokrąglane do najbliższego `0.5`.

## Uruchamianie

Polecenia uruchamiaj z katalogu głównego projektu.

### 1. Trenowanie

NMF:

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_NMF.pkl \
  --alg NMF
```

SVD1:

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_SVD1.pkl \
  --alg SVD1
```

### 2. Predykcja

NMF:

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_NMF.pkl \
  --output_file results/preds_nmf.csv \
  --alg NMF
```

SVD1:

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_SVD1.pkl \
  --output_file results/preds_svd1.csv \
  --alg SVD1
```

### 3. Ewaluacja RMSE (bootstrap)

NMF:

```bash
python -m modules.evaluation.rmse_nmf
```

SVD1:

```bash
python -m modules.evaluation.rmse_svd1
```

Skrypty:
- wykonują `N_RUNS=10` powtórzeń bootstrap,
- trenują i predykują przez `main.py`,
- zapisują metryki do:
  - `results/nmf_rmse_runs.pkl`
  - `results/svd1_rmse_runs.pkl`

## Uwagi

- `main.py` akceptuje: `NMF`, `SVD1`, `SVD2`, `SGD`, `BEST`.
- W `train.py` i `predict.py` działają tylko `NMF` i `SVD1`.
- Dla `SVD2`, `SGD`, `BEST` rzucany jest `NotImplementedError`.
