import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import jaccard_similarity_score # Jaccard Similarity
import math
import implicit
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

from implicit.datasets.movielens import get_movielens

animes = pd.read_excel('E:/Acedemia/DataMining2/Project2/anime.xlsx', header = 0) # load the data
animes['genre'] = animes['genre'].fillna('None') # filling 'empty' data
animes['genre'] = animes['genre'].apply(lambda x: x.split(', ')) # split genre into list of individual genre

genre_data = itertools.chain(*animes['genre'].values.tolist()) # flatten the list
genre_counter = collections.Counter(genre_data)
genres = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})
genres.sort_values('count', ascending=False, inplace=True)

# Plot genre
f, ax = plt.subplots(figsize=(8, 12))
sns.set_color_codes("pastel")
sns.set_style("white")
sns.barplot(x="count", y="genre", data=genres, color='b')
ax.set(ylabel='Genre',xlabel="Anime Count")



users = pd.read_csv('E:/Acedemia/DataMining2/Project2/rating.csv', header = 0)
users.replace({-1: np.nan}, regex=True, inplace = True)
users.head()

megedData = pd.merge(animes,users,on=['anime_id','anime_id'])
megedData1 = megedData[['user_id','name', 'rating_y']]

sparseDf = pd.read_csv('E:/Acedemia/DataMining2/Project2/cross_tab_data.csv', header = 0)

sparseDf= sparseDf[sparseDf.user_id <= 10000]

sparseDf = sparseDf.fillna(0)
sparseDf = sparseDf.drop(['user_id'], axis=1)
sparseDf[sparseDf < 6] = 0
sparseDf[sparseDf > 6] = 1

colNames = list(sparseDf.columns)
colNamesToNumMapper = dict(zip(colNames, range(len(colNames))))
numToColNameMapper = dict(zip(range(len(colNames)),colNames))

sparseDf.columns = range(len(colNames))

#titles, ratings2 = get_movielens('100k')

model = implicit.als.AlternatingLeastSquares(32)
ratings = (bm25_weight(sparseDf,  B=0.9) * 5).tocsr()
ratings = ratings.T.tocsr()

model.fit(ratings)

recommendations = model.recommend(100, ratings,10)

names1 = []
for r in recommendations:
    names1.append(numToColNameMapper[r[0]])

names1.sort()

for n in names1:
    print(n)

related = model.similar_items(3000)


animesDf = pd.read_excel('E:/Acedemia/DataMining2/Project2/anime.xlsx', header = 0)
animesDf = animesDf[animesDf['type']=='TV']
tvAnimieIds = set(animesDf.anime_id.values)
usersDf = pd.read_csv('E:/Acedemia/DataMining2/Project2/rating.csv', header = 0)
usersDf = usersDf[usersDf.anime_id.isin(tvAnimieIds)]


def getGenreComboDict(animesDf):
    
    genreComboDict = {}
    for i,row in animesDf[animesDf.genre.notnull()].iterrows():
        t = tuple(set(sorted(list(row[2].split(',')))))
        if(t not in genreComboDict):
            genreComboDict[t] = []
        genreComboDict[t].append(row[0])
    
    return genreComboDict

def getUserWatchedAnimie(userDf, userId):
    
    userWatched = userDf[userDf.user_id == userId]
    
    animieIdAndRating = []
    for i, row in userWatched[userWatched.rating != -1].iterrows():
        animieIdAndRating.append((row[1], row[2]))
        
    return animieIdAndRating

def getSimilarity(gneres1, geners2):
    s1 = set(gneres1)
    s2 = set(geners2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

#getSimilarity(('a', 'b', 'c'), ('b', 'e', 'd'))

genreComboDict = getGenreComboDict(animesDf)

def getUserRecommendation(userId, genreComboDict, usersDf, animesDf):
    
    ## First get User watched animies
    userWatchedAnimieAndRating = getUserWatchedAnimie(usersDf, userId)

    ## Find the average of the user Rating
    totalRating = 0
    uwAnimieIdsList = []
    for uwa in userWatchedAnimieAndRating:
        totalRating += uwa[1]
        uwAnimieIdsList.append(uwa[0])
        print(animesDf[animesDf.anime_id == uwa[0]].name.values[0])
        
    avgRating = totalRating/len(userWatchedAnimieAndRating)
    
    recList = []
    #Get Userliked animie filter animies with rating >= avg user rating
    userLikedAnimie = []
    print('User liked: ')
    print()
    for uwa in userWatchedAnimieAndRating:
        if(uwa[1] >= avgRating):
           userLikedAnimie.append(uwa)
           print(animesDf[animesDf.anime_id == uwa[0]].name.values[0])
           
    for ula in userLikedAnimie:
        animieId = ula[0]
        geners = list(animesDf[animesDf.anime_id ==  animieId].genre.values[0].split(','))
        geners.sort()
        geners = tuple(set(geners))
        #print(geners)
        genresSimilarity = []
        # get similarity against each genre combo
        for genreCombo in list(genreComboDict.keys()):
            genresSimilarity.append([genreCombo, getSimilarity(genreCombo, geners)])
        
        ## Sort by similarity
        genresSimilarity.sort(key = lambda x: x[1], reverse = True)
        #print(genresSimilarity[0])
        if(len(genreComboDict[genresSimilarity[0][0]])==1):
            dummy = [recList.append(animeId) for animeId in genreComboDict[genresSimilarity[1][0]]]    
        else:
            dummy = [recList.append(animeId) for animeId in genreComboDict[genresSimilarity[0][0]]]
        
    
    recSet = set(recList)
    recSet = recSet - set(uwAnimieIdsList)
    
    recListWithRatings = []
    for rec in recSet:
        recListWithRatings.append([rec,animesDf[animesDf.anime_id == rec].rating.values[0]])
            
    recListWithRatings.sort(key = lambda x: x[1], reverse = True)
    return recListWithRatings

userId = 100
usersRecs = getUserRecommendation(userId, genreComboDict, usersDf, animesDf)

print()
print()
print()
print('Recommendations: ')
for i in range(min(len(usersRecs), 20)):
    print(animesDf[animesDf.anime_id == usersRecs[i][0]].name.values[0])

   