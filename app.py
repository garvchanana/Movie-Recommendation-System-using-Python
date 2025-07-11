import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 🔹 Utility Functions
# ----------------------------



@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies, ratings

@st.cache_data
def build_model(movies, ratings):
    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_item_filled = user_item_matrix.fillna(0)

    # User similarity matrix
    user_similarity = cosine_similarity(user_item_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_filled.index, columns=user_item_filled.index)

    # User mean ratings
    user_means = user_item_matrix.mean(axis=1)

    return user_item_matrix, user_item_filled, user_similarity_df, user_means

def predict_ratings(user_id, user_item_filled, user_similarity_df, user_means):
    sim_scores = user_similarity_df[user_id]
    normalized_ratings = user_item_filled.sub(user_item_filled.mean(axis=1), axis=0)
    weighted_sum = normalized_ratings.T.dot(sim_scores)
    sim_sum = sim_scores.sum()
    predicted = weighted_sum / sim_sum
    predicted += user_means[user_id]
    return predicted

def recommend_movies(user_id, user_item_matrix, user_item_filled, user_similarity_df, user_means, movies, top_n=5):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['title', 'PredictedRating'])

    predicted_ratings = predict_ratings(user_id, user_item_filled, user_similarity_df, user_means)
    seen_movies = user_item_matrix.loc[user_id].dropna().index
    unseen_predictions = predicted_ratings.drop(index=seen_movies)

    top_movie_ids = unseen_predictions.sort_values(ascending=False).head(top_n).index
    top_movies = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]
    top_movies = top_movies.merge(unseen_predictions.rename("PredictedRating"), on='movieId')

    return top_movies.sort_values(by="PredictedRating", ascending=False)

# ----------------------------
# 🔹 Streamlit UI
# ----------------------------

st.title("Movie Recommendation System (UBCF)")
st.write("Built using the MovieLens dataset")

# Load data
movies, ratings = load_data()
user_item_matrix, user_item_filled, user_similarity_df, user_means = build_model(movies, ratings)

# Sidebar - user selection
user_id = st.sidebar.selectbox("Select a User ID", user_item_matrix.index.tolist())

# Number of recommendations
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

# Button to generate recommendations
if st.button("Recommend Movies"):
    recommendations = recommend_movies(
        user_id, user_item_matrix, user_item_filled, user_similarity_df, user_means, movies, top_n=top_n
    )

    if recommendations.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.success(f"Top {top_n} movie recommendations for User {user_id}:")
        st.table(recommendations[['title', 'PredictedRating']].reset_index(drop=True))
