import pandas as pd
from summa import keywords

# Denna kod ändrar på datafilen "keywords.csv".
# Den tar lång tid att köra, så gör inga onödiga ändringar.

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

###################### Dropping unneccessary features #####################

reviews_data.drop(['critic_name'], axis=1, inplace=True)
reviews_data.drop(['top_critic'], axis=1, inplace=True)
reviews_data.drop(['publisher_name'], axis=1, inplace=True)
reviews_data.drop(['review_type'], axis=1, inplace=True)
reviews_data.drop(['review_score'], axis=1, inplace=True)
reviews_data.drop(['review_date'], axis=1, inplace=True)

movies_data = movies # utveckla senare
movies_data.drop(['original_release_date'],axis=1, inplace=True)
movies_data.drop(['streaming_release_date'],axis=1, inplace=True)
movies_data.drop(['runtime'],axis=1, inplace=True)
movies_data.drop(['production_company'],axis=1, inplace=True)
movies_data.drop(['tomatometer_status'],axis=1, inplace=True)
movies_data.drop(['tomatometer_rating'],axis=1, inplace=True)
movies_data.drop(['tomatometer_count'],axis=1, inplace=True)
movies_data.drop(['audience_status'],axis=1, inplace=True)
movies_data.drop(['audience_rating'],axis=1, inplace=True)
#movies_data.drop(['audience_count'],axis=1, inplace=True)
movies_data.drop(['tomatometer_top_critics_count'],axis=1, inplace=True)
movies_data.drop(['tomatometer_fresh_critics_count'],axis=1, inplace=True)
movies_data.drop(['tomatometer_rotten_critics_count'],axis=1, inplace=True)

####################################################################################################

# Slå ihop reviews för samma film.
agg_functions = {'rotten_tomatoes_link': 'first', 'review_content': 'sum', }
combinedReviews = reviews_data.groupby(reviews_data['rotten_tomatoes_link']).aggregate(agg_functions)

# Sätt till samma indexering, för att kunna lägga till kolumn på rätt plats
combinedReviews = combinedReviews.set_index('rotten_tomatoes_link')
movies_data = movies_data.set_index('rotten_tomatoes_link')

revcon = combinedReviews['review_content']
movrev = movies_data.join(revcon)

# före  17712 filmer
# efter 13686 filmer

##################################################################################################

# Keyword extraction (optimized - parallell processing)

import re
from nltk.corpus import stopwords
from concurrent.futures import ProcessPoolExecutor
import logging
import nltk

# Download NLTK stopwords
STOPWORDS = set(stopwords.words('english')) # automatic remove words like “the”, “a”, “an”, or “in”, check: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

# Configure logging - log progress (and errors) in keyword_extraction.log
logging.basicConfig(
    filename='keyword_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Preprocess review text
def preprocess_text(text):
    # Check if text is a string - otherwise convert it to ""
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    # Add other preprocessing steps here (e.g., removing punctuation)
    text = re.sub(r'\W+', ' ', text)
    return text

# Extract keywords with error handling using the nltk keyword-extract (like lab4 part3)
def extract_keywords_safe(review_content):
    try:
        if not review_content.strip():  # Skip empty strings
            return None
        result = keywords.keywords(review_content, ratio=0.2, words=10).replace("\n", " ")
        logging.info(f"Extracted keywords: {result}")   # Write result out in log
        return result
    except Exception as e:  # Error handling
        logging.error(f"Keyword extraction failed: {str(e)}")
        return None

# Parallel processing function - like in lab4 part2B (inspired)
def parallel_keyword_extraction(dataframe, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(extract_keywords_safe, dataframe['cleaned_content']))
    return result

# Ensure multiprocessing code is wrapped correctly (fix for the __main__ problem)
if __name__ == '__main__':

    # # Fake data for test
    # movrev = pd.DataFrame({
    #     'movie_title': ['Movie A', 'Movie B'], 
    #     'review_content': ['This is a great movie', 'I did not like the film'],
    #     'genres': ['action adventure', 'comedy romance']
    # })
    
    # Preprocess review_content
    movrev['cleaned_content'] = movrev['review_content'].apply(preprocess_text)
    
    # TRYING to make the genres on lower case as well
    # movrev['genres'].apply(preprocess_text)

    # Debugging: Cleaned content
    # print("Cleaned Content:")
    # print(movrev['cleaned_content'])

    # Extract keywords in parallel
    movrev['keywords'] = parallel_keyword_extraction(movrev, num_workers=None)  # Use max available cores
    
    # Add genres to keywords
    def combine_keywords_and_genres(row):
        keywords_list = row['keywords'] if row['keywords'] else ""
        genres_list = row['genres'] if isinstance(row['genres'], str) else ""
        return f"{keywords_list} {genres_list}".strip()
    
    # Use function combine_keywords_and_genres
    movrev['combined_keywords'] = movrev.apply(combine_keywords_and_genres, axis=1)
    
    # Drop rows with missing keywords
    movrev = movrev[movrev['combined_keywords'].notnull()]
    
    # Debugging: Check keywords column
    # print("Extracted Keywords:")
    # print(movrev[['review_content', 'keywords']])

    # Drop rows with missing or empty keywords
    movrev = movrev[movrev['keywords'].notnull() & (movrev['keywords'] != "")]

    # Debugging: Check res
    # print("Result:")
    # print(movrev)

    # Save final dataset
    movrev.to_csv('keywords_with_genres.csv', index=False)

    # Print confirmation
    print("keywords_with_genres.csv")
    
    # Observation: 11904 rows (movies) in result