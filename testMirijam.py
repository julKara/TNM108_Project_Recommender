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



# Load the train and test datasets to create two DataFrames
reviews = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
movies = pd.read_csv("data/rotten_tomatoes_movies.csv")

# Keywords
words = pd.read_csv("./keywords.csv")

# User input
userInput = input("What kind of movie do you want to see:")

# test text: I want to see a funny movie in space where the hero travels between planets and fights monsters

# User keywords
userKeywords = keywords.keywords(userInput, ratio=0.8, split=True)

# Collect all user keywords in a single string
temp = ""
for i in userKeywords:
    temp += i + " "

print(temp)

# Turn the string into an array (This workaround was the only way i could get concat to work with all user keywords on a single row)
userString = [temp]

# Add all user keywords at the first row of the dataframe. This makes it easier to calculate cosine similarity.
comp = pd.concat([pd.DataFrame(userString), words['keywords']], ignore_index=True)

# This is for some reason needed to make tf-idf work
comp.columns = ['keywords']

# Convert to tf-idf
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(comp['keywords'])
#print(tfidf_matrix.shape)

# Cosine similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
#print(cos_similarity)

# Find second largest value in cos_similarity. This is the most similar movie since the largest value always is 1 when it compares with itself.
max1 = max2 = float('-inf')
for n in np.nditer(cos_similarity):
    if (n > max1).any():
        max2 = max1
        max1 = n
    elif (n > max2 and n != max1).any():
        max2 = n
        
#print(max2)

# Find the index of the most similar movie. I think the index needs to be adjusted by 1 since a row for user keywords was added.
# So by takin -1 it should match with keywords.csv
index = np.where(np.isclose(cos_similarity, max2))
index = index[1] - 1

#print(index)

# Print out the name of the recommended movie
print(words['movie_title'][index])