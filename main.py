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

# Output
words['Percentage Match'] = csims * 100
words = words.sort_values(by=['Percentage Match'], ascending=False)
words = words[(words['Percentage Match'] > 0.0)]
words = words.set_index('Percentage Match')

# Print the top 10 similar movies with additional context
top_recommendations = words.head(10)

print("\n\n############### Your Review #######################")
print(userInput)
print("############### YOU SHOULD WATCH ##################")
if len(words) != 0:
    for idx, row in top_recommendations.iterrows():
        print(f"{row['movie_title']} | Genres: {row['genres']} | Percentage Match: {idx:.2f}%")
else:
    print("No movies found.")
        
print("\n\n")
