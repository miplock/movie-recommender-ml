# Sample project
### Paweł Lorek

All details are provided in `MCaDR_project1.pdf`.

This example demonstrates how to train a model, generate predictions, and evaluate performance using the provided tools.

---

## 1. Training

The following command:
```
python project1_s123456/main.py --mode train  \
--train_file project1_s123456/data/ratings.csv \
--model_path project1_s123456/models_trained/model_NMF.pkl \
--alg NMF
```


- Reads the `ratings.csv` file
- Trains a simple `NMF` model (with `r=5` and missing values imputed as 0)
- Stores the model (i.e., the approximated matrix `Z`, along with `user_map` and `movie_map`) in a pickle file

---

## 2. Prediction

The following command:

```
python project1_s123456/main.py --mode predict \
--input_file sample_test.csv \
--model_path project1_s123456/models_trained/model_NMF.pkl \
--output_file project1_s123456/results/preds.csv \
--alg NMF
```

- Reads the `sample_test.csv` file (containing `userId` and `movieId`, but no ratings)
- Loads the trained model from `model_NMF.pkl`
- Uses the selected algorithm (`NMF`) to generate predictions
- Stores predictions (with columns `userId,movieId,rating`) in `preds.csv`

---

## 3. Evaluation

The following command:
```
python tools/evaluate_solution.py \
--true_file sample_test_with_ratings.csv \
--pred_file project1_s123456/results/preds.csv
```

- Computes the RMSE between predicted ratings and true ratings
- Uses `sample_test_with_ratings.csv` as ground truth
  (note: this file will not be available during final evaluation)

---

## Notes

- The output file `preds.csv` must follow exactly the same format as in this sample project.
- This sample project implements only the `NMF` algorithm.
- In the full project, you are required to implement additional methods (`SVD1`, `SVD2`, `SGD`, `BEST`).
