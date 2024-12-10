import pandas as pd
import csv
from summa.summarizer import summarize
from summa import keywords
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# 1. Extract keywords from all reviews.
#   a. Add the genre tags.
#   b. Maybe add keywords from the synopsis.
# 2. Input user keywords.
# 3. Match the user keywords to the most similar movies (cosine similarity).
#   a. Factor in popularity.



reviews = pd.read_csv("./data/rotten_tomatoes_critic_reviews.csv")
movies = pd.read_csv("./data/rotten_tomatoes_movies.csv")

# Slå ihop reviews för samma film.
agg_functions = {'rotten_tomatoes_link': 'first', 'review_content': 'sum', }
df_new = reviews.groupby(reviews['rotten_tomatoes_link']).aggregate(agg_functions)
revs = df_new["review_content"]

# print("\n\n")
# print(revs)
# print("\n\n")
# print("Keywords:\n", keywords.keywords(revs["m/star_wars"], words=5))


Z = revs.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0][2])
print(math.degrees(angle_in_radians))




# Skriv ut titel på film
# mask = movies["rotten_tomatoes_link"] == "m/zootopia"
# relRow = movies[mask]
# print("\n\n")
# print(relRow["movie_title"])
