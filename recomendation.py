import pandas as pd
from sklearn.neighbors import NearestNeighbors

df1 = pd.read_csv("/content/Dataset.csv")
df2 = pd.read_csv("/content/Movie_Id_Titles.csv")

df = pd.merge(df1, df2, how="inner", on="item_id")

user_item_matrix = df.pivot_table(index='user_id', columns='title', values='rating', fill_value=0)

k = 5
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(user_item_matrix)

user_index = 0

user_ratings = user_item_matrix.iloc[user_index]

distances, indices = knn_model.kneighbors([user_ratings])

top_n = 10
top_recommendations = indices[0][:top_n]
print(f'Top {top_n} Recommendations for User ID {user_index}:')
for idx in top_recommendations:
    title = user_item_matrix.columns[idx]
    print(f'Item Title: {title}')


'''
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df1 = pd.read_csv("/content/dataset.csv")


df=df1[["id","title","vote_average"]]

user_item_matrix = df.pivot_table(index='id', columns='title', values='vote_average', fill_value=0)

k = 5
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(user_item_matrix)

user_index = 0
print(user_item_matrix.columns[user_index])
user_ratings = user_item_matrix.iloc[user_index]

distances, indices = knn_model.kneighbors([user_ratings])

top_n = 10
top_recommendations = indices[0][:top_n]
print(f'Top {top_n} Recommendations for User ID {user_index}:')
for idx in top_recommendations:
    title = user_item_matrix.columns[idx]
    print(f'Item Title: {title}')
'''