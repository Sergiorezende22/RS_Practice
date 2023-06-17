import pandas as pd
import numpy as np
import math


def cos_similarity(user1_idx, user2_idx, rating_matrix):
    # Pegando avaliações de cada usuários (com avaliações nulas)
    user1_ratings = rating_matrix[user1_idx]
    user2_ratings = rating_matrix[user2_idx]

    # Encontrando indíces de itens avaliados por cada usuário
    user1_valid_ratings_idx = np.where(~np.isnan(user1_ratings))[0]
    user2_valid_ratings_idx = np.where(~np.isnan(user2_ratings))[0]

    # Encontrando indíces de itens coavaliados entre usuários
    co_rated_idx = np.where(~np.isnan(user1_ratings) & ~np.isnan(user2_ratings))[0]

    # Calculando similaridade(trocar nome das variáveis)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for idx in co_rated_idx:
        sum1 += user1_ratings[idx] * user2_ratings[idx]

    for idx in user1_valid_ratings_idx:
        sum2 += user1_ratings[idx] ** 2

    for idx in user2_valid_ratings_idx:
        sum3 += user2_ratings[idx] ** 2

    if (sum2 * sum3) != 0:
        return sum1 / math.sqrt(sum2 * sum3)
    else:
        return 0


def pearson_similarity(user1_idx, user2_idx, rating_matrix):
    # Pegando avaliações de cada usuários (com avaliações nulas)
    user1_ratings = rating_matrix[user1_idx]
    user2_ratings = rating_matrix[user2_idx]

    # Calculando média de avaliação dos usuários
    user1_mean = np.nanmean(user1_ratings)
    user2_mean = np.nanmean(user2_ratings)

    # Encontrando indíces de itens coavaliados entre usuários
    co_rated_idx = np.where(~np.isnan(user1_ratings) & ~np.isnan(user2_ratings))[0]

    # Calculando similaridade(trocar nome das variáveis)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for idx in co_rated_idx:
        sum1 += (user1_ratings[idx] - user1_mean) * (user2_ratings[idx] - user2_mean)
        sum2 += (user1_ratings[idx] - user1_mean) ** 2
        sum3 += (user2_ratings[idx] - user2_mean) ** 2

    if (sum2 * sum3) != 0:
        return sum1 / math.sqrt(sum2 * sum3)
    else:
        return 0


def get_knn(k, similarity_array):

    # Retornando índices dos k maiores valores da similaridade
    return np.argsort(similarity_array)[-k:]


# Abrindo arquivo de avaliações
data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Criando a matriz que relaciona as avaliações (usuários x filmes)
num_users = data['user_id'].nunique()
num_movies = data['movie_id'].nunique()
rating_matrix = np.full((num_users, num_movies), np.nan)

# Preenchendo a matriz com as avaliações dos usuários
for row in data.itertuples(index=False):
    user_idx = row.user_id - 1  # Ajustando o índice do usuário
    movie_idx = row.movie_id - 1  # Ajustando o índice do filme
    rating_matrix[user_idx, movie_idx] = row.rating

pearson_similarity_array = np.full(num_users, 0.0)
cos_similarity_array = np.full(num_users, 0.0)

user_id_test = 1

for idx in range(num_users):
    if idx != user_id_test:
        pearson_similarity_array[idx] = pearson_similarity(user_id_test, idx, rating_matrix)
        cos_similarity_array[idx] = cos_similarity(user_id_test, idx, rating_matrix)

print(get_knn(100, pearson_similarity_array))
print(get_knn(100, cos_similarity_array))

