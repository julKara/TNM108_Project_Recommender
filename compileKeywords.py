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

# detta ger en lista av alla keywords.
# om en film inte kan hitta keywords, ta bort från datan.
# detta tar tid, så vi kör den bara en gång

klist = []
movrev = movrev.reset_index()
i = 0
size = len(movrev)
while (i < len(movrev)):
    try:
        kk = keywords.keywords(movrev.loc[i,'review_content']).replace("\n", " ")
        if kk != "":
            print(i)
            klist.append(kk)
        else:
            print("         " + movrev.loc[i,'movie_title'])
            movrev = movrev.drop(i)
            movrev = movrev.reset_index(drop=True)
            i -= 1
    except: 
        print("         " + movrev.loc[i,'movie_title'])
        movrev = movrev.drop(i)
        movrev = movrev.reset_index(drop=True)
        i -= 1
    i+=1
kseries = pd.Series(klist)
movrev.insert(loc=0, column='keywords', value=kseries)
movrev.to_csv('keywords.csv', index=False)
