import numpy as np
import pandas as pd
import random
from tqdm import tqdm


class RSVD:
    def __init__(self, k=3, learning_rate=0.01, regularization=0.02, iterations=100):
        self.k = k
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.P = None
        self.Q = None
        self.error = float('inf')

    def fit(self, ratings):
        users = max(item[0] for item in ratings) + 1
        items = max(item[1] for item in ratings) + 1

        # Inicializando P e Q caso não estejam inicializadas
        if self.P is None:
            self.P = np.random.rand(users, self.k)
            self.Q = np.random.rand(self.k, items)

        idx = list(range(len(ratings)))
        random.shuffle(idx)

        print(f"Iterando com k igual a {self.k}")
        for _ in tqdm(range(self.iterations)):
            total_error = 0.0

            for i in idx:
                u, v, r = ratings[idx[i]]
                # Cálculo do erro
                e = r - self.predict(u, v)

                # Atualizando variáveis latentes
                P = self.P[u, :]
                Q = self.Q[:, v]
                self.P[u, :] += self.learning_rate * (e * Q - self.regularization * P)
                self.Q[:, v] += self.learning_rate * (e * P - self.regularization * Q)

                total_error += (np.power(e, 2) + self.learning_rate
                                * (np.power(np.linalg.norm(self.Q[:, v]), 2) + np.power(np.linalg.norm(self.P[u, :]),
                                                                                        2)))

            if total_error > self.error:
                break

            self.error = total_error

    def predict(self, user_id, item_id):
        prediction = np.dot(self.P[user_id, :], self.Q[:, item_id])
        return prediction


# Abrindo arquivo de avaliações
data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Criando a matriz que relaciona as avaliações (usuários x filmes)
num_users = data['user_id'].nunique()
num_movies = data['movie_id'].nunique()
ratings_matrix = []

# Preenchendo a matriz com as avaliações dos usuários
for row in data.itertuples(index=False):
    user_idx = row.user_id - 1  # Ajustando o índice do usuário
    movie_idx = row.movie_id - 1  # Ajustando o índice do filme
    ratings_matrix.append([row.user_id, row.movie_id, row.rating])

train = ratings_matrix.copy()

for i in range(1, 10):
    model = RSVD(k=i)
    model.fit(train)
    print(model.error)
