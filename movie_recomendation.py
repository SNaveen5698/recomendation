import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data (replace this with your actual data)
df = pd.read_csv("/content/dataset.csv")

# Select a movie (you can change the index as needed)
selected_movie_index = 2
selected_movie_title = df.loc[selected_movie_index, 'title']


# Clean the overview by removing NaN values
df['overview'] = df['overview'].fillna('')

# Calculate TF-IDF vectors for movie overviews
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['overview'])

# Calculate cosine similarity between the selected movie and all movies
cosine_similarities = cosine_similarity(tfidf_matrix)

# Create a DataFrame with cosine similarities and movie titles
similarities_df = pd.DataFrame(cosine_similarities, columns=df['title'], index=df['title'])

# Get top similar movies (excluding the selected movie itself)
similar_movies = similarities_df[selected_movie_title].sort_values(ascending=False).index[1:11]

print(f"Selected Movie: {selected_movie_title}")
print("Similar Movies:")
print(similar_movies)
