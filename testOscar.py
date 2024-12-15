import pandas as pd
from summa.summarizer import summarize
from summa import keywords

# 1. Extract keywords from all reviews.
#   a. Add the genre tags.
#   b. Maybe add keywords from the synopsis.
# 2. Input user keywords.
# 3. Match the user keywords to the most similar movies (cosine similarity).
#   a. Factor in popularity.

keys = pd.read_csv("./keywords_with_genres.csv")

#print(keys["genres"].to_string())

kh = keys["review_content"] # jämför med reviews 
kh = keys["keywords"]       # jämför med keywords från reviews... mycket sämre
inputkey = "Amazing music"

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

#csims = '{:.1%}'.format(csims)

keys['Percentage Match'] = csims * 100

keys = keys.sort_values(by=['Percentage Match'], ascending=False)
keys = keys[(keys['Percentage Match'] > 0.0)]
keys = keys.set_index('Percentage Match')
keys = keys.head(10)

print("\n\n############### IF YOU WANT #######################")
print(inputkey)
print("############### YOU SHOULD WATCH ##################")
if len(keys) != 0:
    print(keys['movie_title'])
else:
    print("Nothing")
print("\n\n")
