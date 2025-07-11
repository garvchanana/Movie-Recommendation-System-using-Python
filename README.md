#  Movie Recommendation System (User-Based Collaborative Filtering)

This project is a web-based movie recommendation system built using **Python** and **Streamlit**, leveraging the **MovieLens public dataset**. The system uses **User-Based Collaborative Filtering (UBCF)** to recommend personalized movie suggestions based on similar users' preferences.

---

## Introduction

Recommendation systems are crucial in helping users discover relevant content in massive datasets. In this project, we focus on recommending movies to users based on the **ratings of other users with similar tastes**.

The system uses the **cosine similarity** measure between user vectors in a user-item matrix to predict movie ratings for unseen movies and recommend the top-rated ones.

---

## Features

- 🔢 Select any user ID from the dataset
- 🎯 Predict ratings for movies the user hasn't seen
- 🎬 Display **Top-N Recommended Movies** with predicted scores
- 🧹 Cleaned and reformatted movie titles for better readability
- 📊 (Optional) EDA features like genre-wise ratings and tag distribution

---

## How It Works

1. Load and preprocess MovieLens dataset (movies.csv, ratings.csv)
2. Build a **user-item matrix** from rating data
3. Compute **user-to-user cosine similarity**
4. Predict ratings for unseen movies using weighted average of similar users
5. Recommend top N movies with highest predicted ratings

---

## Tech Stack

| Technology      | Description                                      |
|-----------------|--------------------------------------------------|
| 🐍 Python       | Programming language                             |
| 📊 Pandas       | Data manipulation and analysis                   |
| 🎯 scikit-learn | Cosine similarity computation                    |
| 🎈 Streamlit    | Web app development framework                    |
| 📁 MovieLens    | Dataset used (movies.csv, ratings.csv,links.csv,tags.csv)           |

