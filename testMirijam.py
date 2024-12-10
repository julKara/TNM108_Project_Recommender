# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from summa import keywords


# Load the train and test datasets to create two DataFrames

reviews = pd.read_csv("./rotten_tomatoes_critic_reviews.csv")

movies = pd.read_csv("./rotten_tomatoes_movies.csv")

index = 1

agg_functions = {'rotten_tomatoes_link': 'first', 'review_content': 'sum', }
df_new = reviews.groupby(reviews['rotten_tomatoes_link']).aggregate(agg_functions)
revText = df_new["review_content"]

print(len(revText))

print(revText.iloc(index))

#print(df_new['rotten_tomatoes_link'][index])

#print("Keywords:\n",keywords.keywords(revText[index]))

#id = df_new['rotten_tomatoes_link'][index] == movies['rotten_tomatoes_link']

#print(movies['movie_title'][id])