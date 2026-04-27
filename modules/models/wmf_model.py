import numpy as np
import pandas as pd


class WMFRecommender:
    def __init__(self, n_factors=20, lr=0.01, reg=0.02, n_epochs=20, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        self.user_to_index = None
        self.movie_to_index = None

        self.U = None
        self.V = None

    def fit(self, ratings_df: pd.DataFrame):

        np.random.seed(self.seed)

        users = ratings_df["userId"].unique()
        items = ratings_df["movieId"].unique()

        self.user_to_index = {u: i for i, u in enumerate(users)}
        self.movie_to_index = {m: j for j, m in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        # latent factors
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        for epoch in range(self.n_epochs):
            total_loss = 0

            for row in ratings_df.itertuples(index=False):
                u = self.user_to_index[row.userId]
                i = self.movie_to_index[row.movieId]
                r = row.rating

                pred = np.dot(self.U[u], self.V[i])
                err = r - pred

                # SGD update
                u_grad = err * self.V[i] - self.reg * self.U[u]
                i_grad = err * self.U[u] - self.reg * self.V[i]

                self.U[u] += self.lr * u_grad
                self.V[i] += self.lr * i_grad

                total_loss += err ** 2

            print(f"Epoch {epoch}: loss={total_loss:.4f}")

        return self

    def _predict_known_pair(self, user_id, movie_id):
        if user_id not in self.user_to_index or movie_id not in self.movie_to_index:
            return 3.0  # fallback global mean

        u = self.user_to_index[user_id]
        i = self.movie_to_index[movie_id]

        return float(np.dot(self.U[u], self.V[i]))
    