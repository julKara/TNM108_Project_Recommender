import pandas as pd
from summa.summarizer import summarize
from summa import keywords

# 1. Extract keywords from all reviews.
#   a. Add the genre tags.
#   b. Maybe add keywords from the synopsis.
# 2. Input user keywords.
# 3. Match the user keywords to the most similar movies (cosine similarity).
#   a. Factor in popularity.

###################### Importing and preprocessing data #####################

# Ladda in data-filerna.
reviews = pd.read_csv("./data/rotten_tomatoes_critic_reviews.csv")
movies = pd.read_csv("./data/rotten_tomatoes_movies.csv")

# Filter data where the content of the review is not null
data = reviews[reviews['review_content'].notnull()]

data = reviews[reviews['review_score'].notnull()]

# Seperate fresh and rotten reviews
fresh_reviews = data[(data['review_type'] == 'Fresh') & (data['review_score'].str.match(r'\d+/\d+'))]
rotten_reviews = data[(data['review_type'] == 'Rotten')& (data['review_score'].str.match(r'\d+/\d+'))]

# Randomly Sample 75000 datapoints each to make up a 'balanced' dataset of both Fresh and Rotten at 50:50 ratio
sampled_fresh = fresh_reviews.sample(n=75000, random_state=42) 
sampled_rotten = rotten_reviews.sample(n=75000, random_state=42)

# Combine both sampled data
reviews_data = pd.concat([sampled_fresh, sampled_rotten])

#Shuffle and make the final dataset
reviews_data = reviews_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Look at the top few rows of the dataset
# print(reviews_data.head())

# Print the data types
# print(reviews_data.dtypes)

###################### Dropping unneccessary features #####################
reviews_data.drop(['critic_name'], axis=1, inplace=True)
reviews_data.drop(['top_critic'], axis=1, inplace=True)
reviews_data.drop(['publisher_name'], axis=1, inplace=True)
reviews_data.drop(['review_type'], axis=1, inplace=True)
reviews_data.drop(['review_score'], axis=1, inplace=True)
reviews_data.drop(['review_date'], axis=1, inplace=True)

# Print the remaining data types
# print(reviews_data.dtypes)

# Slå ihop reviews för samma film.
agg_functions = {'rotten_tomatoes_link': 'first', 'review_content': 'sum', }
df_new = reviews_data.groupby(reviews_data['rotten_tomatoes_link']).aggregate(agg_functions)
combinedReviews = df_new["review_content"]

# How much we have left
print(len(combinedReviews))

df = data[['rotten_tomatoes_link', 'review_content']][:20]
print(df.head())