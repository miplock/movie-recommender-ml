# Movie Recommender System (Project 1)

This project implements a command-line movie recommender workflow based on
the requirements from `2025_26_MCaDR_project1.pdf`.

Current implementation supports:
- training a model (`--mode train`)
- generating predictions from a saved model (`--mode predict`)
- the `NMF` algorithm

Algorithms listed in the assignment but not implemented yet in this code:
`SVD1`, `SVD2`, `SGD`, `BEST`.

## Project Structure

```text
project1_s324671/
├── data/
│   └── ratings.csv
├── modules/
│   ├── models/
│   │   └── nmf_model.py
│   ├── train.py
│   └── predict.py
├── models_trained/
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Data Format

Training input (`ratings.csv`) should contain at least:
- `userId`
- `movieId`
- `rating`

Prediction input should contain:
- `userId`
- `movieId`

Prediction output (`preds.csv`) contains:
- `userId`
- `movieId`
- `rating`

Predicted ratings are clipped to `[0.5, 5.0]` and rounded to the nearest
`0.5`.

## How To Run

Run commands from the project root directory.

### 1. Train

```bash
python main.py --mode train \
  --train_file data/ratings.csv \
  --model_path models_trained/model_NMF.pkl \
  --alg NMF
```

What happens:
- training CSV is loaded
- NMF model is trained (`n_components=15`,
  `imputation_strategy="movie_mean"`)
- model is saved to `--model_path` (pickle)

### 2. Predict

```bash
python main.py --mode predict \
  --input_file sample_test.csv \
  --model_path models_trained/model_NMF.pkl \
  --output_file results/preds.csv \
  --alg NMF
```

What happens:
- saved model is loaded from `--model_path`
- input pairs (`userId`, `movieId`) are read from `--input_file`
- predictions are generated and written to `--output_file`

## Implemented NMF Details

The NMF pipeline is implemented in `modules/models/nmf_model.py` using
`sklearn.decomposition.NMF`.

Main steps:
- build sparse user-movie matrix from input ratings
- impute missing values (`zero`, `movie_mean`, or `user_mean`)
- factorize matrix into latent factors `W` and `H`
- predict by matrix reconstruction (`W @ H`)
- handle cold-start cases with fallback:
  movie mean -> user mean -> global mean
- clip and round ratings to assignment-compatible scale

## Notes

- `main.py` accepts assignment-required algorithm names:
  `NMF`, `SVD1`, `SVD2`, `SGD`, `BEST`.
- `train.py` and `predict.py` currently execute only `NMF`.
- For non-`NMF` selection, the code raises `NotImplementedError`.
