import pandas as pd
from summa.summarizer import summarize
from summa import keywords

# 1. Extract keywords from all reviews.
#   a. Add the genre tags.
#   b. Maybe add keywords from the synopsis.
# 2. Input user keywords.
# 3. Match the user keywords to the most similar movies (cosine similarity).
#   a. Factor in popularity.


# Ladda in data-filerna.
reviews = pd.read_csv("./data/rotten_tomatoes_critic_reviews.csv")
movies = pd.read_csv("./data/rotten_tomatoes_movies.csv")

# Slå ihop reviews för samma film.
agg_functions = {'rotten_tomatoes_link': 'first', 'review_content': 'sum', }
df_new = reviews.groupby(reviews['rotten_tomatoes_link']).aggregate(agg_functions)
combinedReviews = df_new["review_content"]
