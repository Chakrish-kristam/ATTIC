import streamlit as st
import pandas as pd
import os
import joblib
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
import string
import re
import requests

# Load the model and data
# directory = "saved_model"
# file_path = os.path.join(directory, "nearest_neighbors.joblib")
nn = joblib.load(nearest_neighbors.joblib)

# Read the CSV file using pandas
movies = pd.read_csv("moviesdatabase.csv", encoding='ISO-8859-1')
movies = movies.dropna(subset=["reviews"])

# Set up TextVectorization and Embedding layers
max_vocab_length = 10000
max_length = 1000
text_vectorizer = TextVectorization(
    max_tokens=max_vocab_length,
    output_mode="int",
    output_sequence_length=max_length,
    pad_to_max_tokens=True
)
text_vectorizer.adapt(movies["reviews"])

embedding = layers.Embedding(
    input_dim=max_vocab_length,
    output_dim=32,
    input_length=max_length
)




# Create a Streamlit web app
def main():
    st.title("Movie Recommendation System")

    movie_input = st.text_input("Enter a movie review:")
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(movie_input)
        display_recommendations(recommendations)

def get_recommendations(input_review):
    input_vector = embedding(text_vectorizer(input_review))
    neighbors = nn.kneighbors(input_vector, return_distance=False)
    return neighbors[0]

def display_recommendations(recommendations):
    st.subheader("Recommended Movies:")
    for index in recommendations:
        movie_info = movies.iloc[index-1]
        st.write(f"Movie: {movie_info['title']} - Genre: {movie_info['genre']}")
        st.image(movie_info['poster_path'],width=150)

if __name__ == "__main__":
    main()
