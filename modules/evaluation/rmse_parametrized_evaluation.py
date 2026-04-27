import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from modules.models.svd2_model import SVD2Recommender
from modules.models.svd1_model import SVD1Recommender
from modules.models.sgd_model import SGDRecommender
from modules.models.nmf_model import NMFRecommender
from modules.models.wmf_model import WMFRecommender

def evaluate_rmse(
    model_name: str,
    strategy: str,
    ratings_df: pd.DataFrame,
    components_list: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
    output_path: str = "svd2_results.csv",
) -> pd.DataFrame:

    results = []

    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state,
    )

    for n in components_list:
        print(f"\n=== Evaluating n_components={n}, model: {model_name} ===")

        if model_name == 'SVD2':
            model = SVD2Recommender(
                n_components=n,
                imputation_strategy=strategy,
                random_state=random_state,
                n_iters=100,
                tol=1e-8,
            )
        elif model_name == 'SVD1':
            model = SVD1Recommender(
                n_components=n,
                imputation_strategy=strategy
            )
        elif model_name == 'NMF':
            model = NMFRecommender(
                n_components=n,
                imputation_strategy=strategy
            )
        elif model_name == 'WMF':
            model = WMFRecommender(
                n_factors=n
            )
        elif model_name == 'SGD':
            model = SGDRecommender(
                rank = n
            )
        else:
            raise NotImplementedError(
            f"Algorithm {model_name} is not implemented"
        )           
        
        model.fit(train_df)
       
        y_true = []
        y_pred = []

        for row in test_df.itertuples(index=False):
            user_id = row.userId
            movie_id = row.movieId
            true_rating = row.rating

            
            if user_id not in model.user_to_index:
                continue
            if movie_id not in model.movie_to_index:
                continue

            pred = model._predict_known_pair(user_id, movie_id)

            y_true.append(true_rating)
            y_pred.append(pred)

        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"RMSE: {rmse:.4f}")

        results.append({
            "n_components": n,
            "rmse": rmse,
            "n_predictions": len(y_true),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\nWyniki zapisane do: {output_path}")

    return results_df



if __name__ == "__main__":
    ratings = pd.read_csv("data/ratings.csv")  

    components = list(range(1, 26, 2))
    
    models = ['NMF', 'SVD1', 'SVD2', 'SGD', 'WMF']
    strats = ['zero', 'movie_mean', 'user_mean']


    for model_n in models:
        for strat in strats:     
            if model_n in ['SGD', 'WMF']:   # SGD i WMF nie są parametryzowalne przez
                if strat != 'zero':         # strat, więc robimy pojedynczą ewaluację
                    break 
                else:
                    strat = 'none'       
            evaluate_rmse(
                model_name = model_n,                    
                strategy = strat,                       
                ratings_df=ratings,
                components_list=components,
                test_size=0.2,
                output_path=f"parametrized_rmse_evaluation_results/{model_n}_{strat}_rmse_results.csv",
            )