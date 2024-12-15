# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from summa import keywords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords



# Keywords
words = pd.read_csv("./keywords_with_genres.csv")

# User input
userInput = input("Write a review: ")

while userInput == "":
    userInput = input("Please write a review: ")


# test text: I want to see a funny movie in space where the hero travels between planets and fights monsters

# Turn the string into an array (Needed for tfidf)
userString = [userInput]

# Convert to tf-idf
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
keyword_matrix = tfidf_vectorizer.fit_transform(words['combined_keywords'])
input_tfidf = tfidf_vectorizer.transform(userString)

# Cosine similarity
cos_similarity = cosine_similarity(input_tfidf, keyword_matrix)
csims = cos_similarity[0]

# Apply similarity score to dataset
words['score'] = csims
words = words.sort_values(by=['score'], ascending=False)
words = words.set_index('score')

# Print 10 most similar movies
print(words.head(10)['movie_title'])