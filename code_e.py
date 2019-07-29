#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:05:39 2019

@author: eswarchand
"""

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator
import matplotlib as mlb

mlb.axes.plot
import matplotlib.pyplot as plt

anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')
rating1 = pd.read_csv('rating.csv')

# No. of distinct Anime
len(anime.anime_id.unique())

len(rating.user_id.unique())


# Before alteration the ratings dataset uses a "-1" to represent missing ratings. 
# I'm replacing these placeholders with a null value because 
# I will later be calculating the average rating per user and don't want the average to be distorted

rating.rating.replace({-1: np.nan}, regex=True, inplace = True)
rating.head()

anime_tv = anime[anime['type']=='TV']
anime_tv.head()


# Join the two dataframes on the anime_id columns

merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)

# For computing reasons I'm limiting the dataframe length to 10,000 users

merged=merged[['user_id', 'name', 'user_rating']]
merged_sub= merged[merged.user_id <= 10000]
merged_sub.head()

# Pivoting table to use for collaborative filtering
#piv_all=merged.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
piv = merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
piv_all.to_csv('cross_tab_data.csv')
          
# Note: As we are subtracting the mean from each rating to standardize
# all users with only one rating or who had rated everything the same will be dropped

# Normalize the values
piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)


# Drop all columns containing only zeros representing users who did not rate
piv_norm.fillna(0, inplace=True)
piv_norm = piv_norm.T
piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]



# Our data needs to be in a sparse matrix format to be read by the following functions

piv_sparse = sp.sparse.csr_matrix(piv_norm.values)



item_similarity = cosine_similarity(piv_sparse)
user_similarity = cosine_similarity(piv_sparse.T)



# Inserting the similarity matricies into dataframe objects

item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)
user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)



# This function will return the top 10 shows with the highest cosine similarity value

def top_animes(anime_name):
    count = 1
    print('Similar shows to {} include:\n'.format(anime_name))
    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:
        print('No. {}: {}'.format(count, item))
        count +=1  

# This function will return the top 5 users with the highest similarity value 

def top_users(user):
    
    if user not in piv_norm.columns:
        return('No data available on user {}'.format(user))
    
    print('Most Similar Users:\n')
    sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:11]
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    zipped = zip(sim_users, sim_values,)
    for user, sim in zipped:
        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim)) 

# This function constructs a list of lists containing the highest rated shows per similar user
# and returns the name of the show along with the frequency it appears in the list

uwAnimieIdsList = []
    for uwa in userWatchedAnimieAndRating:
        totalRating += uwa[1]
        uwAnimieIdsList.append(uwa[0])
recSet = set(recList)
    recSet = recSet - set(uwAnimieIdsList)

def similar_user_recs(user):
    
    if user not in piv_norm.columns:
        return('No data available on user {}'.format(user))
    
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    best = []
    most_common = {}
    
    for i in sim_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:15]

# This function calculates the weighted average of similar users
# to determine a potential rating for an input user and show

def predicted_rating(anime_name, user):
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:1000]
    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]
    rating_list = []
    weight_list = []
    for j, i in enumerate(sim_users):
        rating = piv.loc[i, anime_name]
        similarity = user_values[j]
        if np.isnan(rating):
            continue
        elif not np.isnan(rating):
            rating_list.append(rating*similarity)
            weight_list.append(similarity)
    return sum(rating_list)/sum(weight_list) 

# Testing the functions
    
top_animes('Steins;Gate')

top_users(3)

userId = 100
recs = similar_user_recs(userId)
recNamesSet = set([rec[0] for rec in recs])

ruwAnimeIds = set(rating[rating.user_id == userId].anime_id.values)
watchednamesSet = set(anime_tv[anime_tv.anime_id.isin(ruwAnimeIds)].name.values)

print(recNamesSet - watchednamesSet)

for rec in sorted(list(recNamesSet - watchednamesSet)):
    print(rec)

predicted_rating('Boku dake ga Inai Machi',3)

# Creates a list of every show watched by user 3

watched = piv.T[piv.loc[3,:]>0].index.tolist()

# Make a list of the squared errors between actual and predicted value

errors = []
for i in watched:
    actual=piv.loc[3, i]
    predicted = predicted_rating(i, 3)
    errors.append((actual-predicted)**2)
    
np.mean(errors)



######### EDA #################

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import collections
import operator

anime = anime.replace({'Harem': 'Hentai'}, regex=True)
anime = anime.replace({'Ecchi': 'Hentai'}, regex=True)
anime = anime.replace({'Shoujo Ai': 'Hentai'}, regex=True)
anime = anime.replace({'Yaoi': 'Hentai'}, regex=True)
anime = anime.replace({'Yuri': 'Hentai'}, regex=True)
anime = anime.replace({'Shounen Ai': 'Hentai'}, regex=True)

anime = anime.replace({'Demons': 'Vampire'}, regex=True)

anime = anime.replace({'Supernatural': 'Magic'}, regex=True)
anime = anime.replace({'Super Power': 'Magic'}, regex=True)
anime = anime.replace({'Sci-Fi': 'Magic'}, regex=True)

genres = set()
for entry in anime['genre']:
    if not type(entry) is str:
        continue
    genres.update(entry.split(", "))
print(genres)
print("Total Genres: " + str(len(genres)))

# List genres by count
genres_count = collections.defaultdict(int)
for entry in anime['genre']:
    if not type(entry) is str:
        continue
    seen_already = set()
    for genre in entry.split(", "):
        if genre in seen_already:
            continue
        seen_already.add(genre)
        genres_count[genre] += 1
sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)


genres_count_df = pd.DataFrame.from_dict(genres_count, orient='index').reset_index()
genres_count_df.columns = ['genre','animes']

# Plot all animes by rating and popularity colored by genre
fig = plt.figure(figsize=(20,20))
ax = plt.gca()
plt.title('All Animes Rating vs. Popularity By Genre')
plt.xlabel('Rating')
plt.ylabel('Popularity (People)')
num_colors = len(genres)
cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])
ax.set_yscale('log')
#ax.plot(genres_count_df["genre"], genres_count_df['animes'], marker='o', linestyle='', ms=12, label=genre)

# For each genre, plot data point if it falls in that category
for genre in genres:
    data_genre = anime[anime.genre.str.contains(genre) == True]
    ax.plot(genres_count_df["genre"], genres_count_df['animes'], marker='o', linestyle='', ms=12, label=genre)
ax.legend(numpoints=1, loc='upper left');



fig = plt.figure(figsize=(5,5))
ax = plt.gca()
plt.title('Genre distribution by No of Anime')
plt.xlabel('Genre')
plt.ylabel('No. of Anime')
num_colors = 5
cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])
ax.set_yscale('log')
ax.plot(genres_count_df["genre"], genres_count_df['animes'], marker='o', linestyle='', ms=12, label=genre)

fig = plt.figure(figsize=(20,20))
ax = plt.gca()
plt.title('Genre distribution by No of Anime')
plt.xlabel('Genre')
plt.ylabel('No. of Anime')
plt.plot(genres_count_df["genre"], genres_count_df['animes'], color='green', marker='o', linestyle='dashed',
linewidth=2, markersize=12)

genres_members_df = pd.DataFrame()
genres_members_df.columns = ['genre','animes']

members=[]
for genre in genres:
    data_genre = anime[anime.genre.str.contains(genre) == True]
    members+=anime['members']
    

genres_members_df = pd.DataFrame.from_dict(genres_members, orient='index').reset_index()
genres_members_df.columns = ['genre','animes']

genres_members_df = genres_members_df.sort_values('animes',ascending=False)



################# For EDA #########################

anime = pd.read_csv('anime.csv')
# filling 'empty' data
anime['genre'] = anime['genre'].fillna('None') 
# split genre into list of individual genre
anime['genre'] = anime['genre'].apply(lambda x: x.split(', ')) 

# flatten the list
genre_data = itertools.chain(*anime['genre'].values.tolist()) 
genre_counter = collections.Counter(genre_data)
genres1 = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})
genres1.sort_values('count', ascending=False, inplace=True)



for i,row in anime.iterrows():
    #print(row)
    #print(row[2])
    genres1 = [genre.strip() for genre in row[2]]
    
    for genre in genres:
        if(genre in genreRatingsDict):
            if(not math.isnan(row[5])):
                genreRatingsDict[genre].append(float(row[5]))
            if(row[4] != 'Unknown'):
                genreEpisodesDict[genre].append(float(row[4]))
            genereAnimieCountDict[genre] += 1
            genreViews[genre].append(float(row[6]))
        else:
            if(not math.isnan(row[5])):
                genreRatingsDict[genre] = []
                genreRatingsDict[genre].append(float(row[5]))
            if(row[4] != 'Unknown'):
                genreEpisodesDict[genre] = []
                genreEpisodesDict[genre].append(float(row[4]))
            genereAnimieCountDict[genre] = 1
            genreViews[genre] = []
            genreViews[genre].append(float(row[6]))
    

genreRatingsDict_sorted = sorted(genreRatingsDict.items(), key=operator.itemgetter(1))


