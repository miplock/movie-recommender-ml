# Movie Recommender System (Project 1)
### Mikołaj (s324671)
### Tomasz (s337669)

Implementacja projektu 1 z MCaDR oparta o interfejs CLI w `main.py`.

Aktualnie zaimplementowane:
- trenowanie modelu (`--mode train`)
- predykcja ocen (`--mode predict`)
- algorytmy: `NMF`, `SVD1`, `SVD2`, `SGD`, `BEST`
- skrypty ewaluacji RMSE (bootstrap):
  - `modules/evaluation/rmse_nmf.py`
  - `modules/evaluation/rmse_svd1.py`
  - `modules/evaluation/rmse_svd2.py`
  - `modules/evaluation/rmse_sgd.py`

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
│   │   ├── sgd_model.py
│   │   └── svd1_model.py
│   │   └── svd2_model.py
│   ├── evaluation/
│   │   ├── rmse_common.py
│   │   ├── rmse_nmf.py
│   │   └── rmse_svd1.py
│   │   └── rmse_svd2.py
│   │   └── rmse_sgd.py
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

SVD2:

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_SVD2.pkl \
  --alg SVD2
```

SGD:

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_SGD.pkl \
  --alg SGD
```

BEST:

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_BEST.pkl \
  --alg BEST
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

SVD2:

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_SVD2.pkl \
  --output_file results/preds_svd2.csv \
  --alg SVD2
```

SGD:

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_SGD.pkl \
  --output_file results/preds_sgd.csv \
  --alg SGD
```

BEST:

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_BEST.pkl \
  --output_file results/preds_best.csv \
  --alg BEST
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

SVD2:

```bash
python -m modules.evaluation.rmse_svd2
```

SGD:

```bash
python -m modules.evaluation.rmse_sgd
```

BEST:

```bash
python -m modules.evaluation.rmse_best
```

Skrypty:
- wykonują `N_RUNS=10` powtórzeń bootstrap,
- trenują i predykują przez `main.py`,
- zapisują metryki do:
  - `results/nmf_rmse_runs.pkl`
  - `results/svd1_rmse_runs.pkl`
  - `results/svd2_rmse_runs.pkl`
  - `results/sgd_rmse_runs.pkl`
  - `results/best_rmse_runs.pkl`
 
