from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import random
from collections import Counter
import itertools

# Initialize Flask app
app = Flask(__name__)

# Load model and encoders
model = load_model("models/model1.h5")
user_enc = joblib.load('user_encoder.pkl')
movie_enc = joblib.load('movie_encoder.pkl')

# Load datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Ensure all titles are strings
movies['title'] = movies['title'].astype(str)

# Merge for user-rated info
data = pd.merge(ratings, movies, on="movieId")
data['user'] = user_enc.transform(data['userId'])
data['movie'] = movie_enc.transform(data['movieId'])

# Dictionaries for mapping
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
movie_id_to_genre = dict(zip(movies['movieId'], movies['genres']))

def recommend(user_input_id):
    try:
        user_input_id = int(user_input_id)
        if user_input_id not in user_enc.classes_:
            return None, None, None

        encoded_user = user_enc.transform([user_input_id])[0]
        rated_movies_df = ratings[ratings['userId'] == user_input_id][['movieId', 'rating']]

        if rated_movies_df.empty:
            return None, None, None

        rated = pd.merge(rated_movies_df, movies[['movieId', 'title', 'genres']], on='movieId')[['title', 'genres', 'rating']]

        rated_movie_ids = rated_movies_df['movieId'].unique()
        all_movie_ids = movies['movieId'].unique()
        unrated_movie_ids = list(set(all_movie_ids) - set(rated_movie_ids))
        valid_movies = [mid for mid in unrated_movie_ids if mid in movie_enc.classes_]

        if not valid_movies:
            return rated, [], []

        sampled_movies = random.sample(valid_movies, min(10, len(valid_movies)))
        encoded_movies = movie_enc.transform(sampled_movies)

        user_input = np.full((len(encoded_movies),), encoded_user).reshape(-1, 1)
        encoded_movies = np.array(encoded_movies).reshape(-1, 1)

        preds = model.predict([user_input, encoded_movies], verbose=0)
        top_indices = preds.flatten().argsort()[::-1][:5]
        top_movie_ids = [sampled_movies[i] for i in top_indices]
        top_movies = [(movie_id_to_title[mid], movie_id_to_genre[mid]) for mid in top_movie_ids]

        high_rated = rated[rated['rating'] >= 4]['genres']
        genre_counts = Counter(itertools.chain.from_iterable(g.split('|') for g in high_rated))
        top_genres = [g for g, _ in genre_counts.most_common(3)]

        content_rec = movies[
            movies['genres'].apply(lambda g: any(tg in g for tg in top_genres)) &
            (~movies['movieId'].isin(rated_movie_ids))
        ].head(5)

        content_movies = list(zip(content_rec['title'], content_rec['genres']))

        return rated, top_movies, content_movies

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        rated, recommendations, content_rec = recommend(user_id)

        if rated is None:
            return render_template('index.html', error="Invalid or unknown User ID.")

        return render_template('index.html', user_id=user_id, rated=rated.itertuples(), recommendations=recommendations, content_rec=content_rec)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
