import pandas as pd
from summa.summarizer import summarize
from summa import keywords

# 1. Extract keywords from all reviews.
#   a. Add the genre tags.
#   b. Maybe add keywords from the synopsis.
# 2. Input user keywords.
# 3. Match the user keywords to the most similar movies (cosine similarity).
#   a. Factor in popularity.

keys = pd.read_csv("./keywords.csv")

#print(keys["genres"].to_string())

kh = keys["review_content"] # jämför med reviews 
#kh = keys["keywords"]       # jämför med keywords från reviews... mycket sämre
inputkey = "haunted house"

Z = kh
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
from sklearn.metrics.pairwise import cosine_similarity
iktfidf = tfidf_vectorizer.transform([inputkey])
cos_similarity = cosine_similarity(iktfidf[0], tfidf_matrix)
csims = cos_similarity[0]

popularity = keys["audience_count"].to_numpy()
#csims = csims + (popularity / 100000000) # delat på en miljard
#csims = csims * popularity

keys['score'] = csims
keys = keys.sort_values(by=['score'], ascending=False)
keys = keys.set_index('score')


print("\n\n############### IF YOU WANT #######################")
print(inputkey)
print("############### YOU SHOULD WATCH ##################")
print(keys.head(10)['movie_title'])
print("\n\n")
