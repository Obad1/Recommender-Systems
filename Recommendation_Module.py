#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import os

base_dir = os.path.dirname(__file__)

tmdb_data_path = os.path.join(base_dir, 'data', 'tmdb_movies_data.csv')
small_data_path = os.path.join(base_dir, 'data', 'small_movie_dataset.csv')
movie_dataset_path = os.path.join(base_dir, 'data', 'movie_metadata.csv')
large_data_path = os.path.join(base_dir, 'data', 'updated_movie_dataset_with_user_id.csv')


tmdb_data = pd.read_csv(tmdb_data_path)
small_data = pd.read_csv(small_data_path)
movie_dataset = pd.read_csv(movie_dataset_path)
large_data = pd.read_csv(large_data_path)


first_merge= pd.merge(tmdb_data, movie_dataset, on =['title','director','genre'],how='right')

first_merge = first_merge[['title','director','genre']]

second_merge = pd.merge(small_data,large_data,how='outer')

second_merge = pd.merge(first_merge,second_merge,how='outer')

second_merge['final_movie_title'] = second_merge['title'].combine_first(second_merge['movie_name'])

second_merge = second_merge.drop(columns=['title','movie_name'])

second_merge.loc[:,'rating']=second_merge['rating'].fillna(second_merge['rating'].mean())

second_merge.loc[:,'year_movie_was_created']=second_merge['year_movie_was_created'].fillna(second_merge['year_movie_was_created'].mean()).round(4)

second_merge = second_merge.dropna(subset=['user_id'])

user_ids = second_merge['user_id']

second_merge.loc[:,'user_id'] = second_merge['user_id'].astype('int')
ratings = second_merge['rating'].values

# Step 2: Perform one-hot encoding on categorical variables(letters) and normalization on numerical variables(numbers)

one_hot_encoded = pd.get_dummies(second_merge,columns=['genre','director','final_movie_title'],drop_first=True)

# PERFORM NORMALIZATION

from sklearn.preprocessing import MinMaxScaler  

scaler = MinMaxScaler()

one_hot_encoded[['rating','year_movie_was_created','user_id']] = scaler.fit_transform(one_hot_encoded[['rating','year_movie_was_created','user_id']])

one_hot_encoded.tail()


from sklearn.model_selection import train_test_split

X = pd.get_dummies(second_merge.drop(columns=['rating']),columns=['genre','director','final_movie_title'],drop_first=True)
y = second_merge['rating'].values

X_train_val,y_train_val,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=46)

# Make sure both have the same length by truncating the extra samples
min_length = min(X_train_val.shape[0], y_train_val.shape[0])
X_train_val = X_train_val[:min_length]
y_train_val = y_train_val[:min_length]

#ensure both are the same sha
assert X_train_val.shape[0] == y_train_val.shape[0]

X_train,X_cross_eval,y_train,y_cross_eval = train_test_split(X_train_val,y_train_val,test_size=0.25,random_state= 95)

y_series = pd.Series(y)  

import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Concatenate,Embedding,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder 

num_movies = second_merge['final_movie_title'].nunique()
num_users = second_merge['user_id'].nunique()

user_input = Input(shape=(1,),name='user_input')
movie_input = Input(shape=(1,),name='movie_input')

max_user_id = int(second_merge['user_id'].max() + 1)
second_merge.loc[:,'movie_id'] = second_merge['final_movie_title'].factorize()[0]
max_movie_id = int(second_merge['movie_id'].max() + 1)

# Calculate the number of users
num_users = int(second_merge['user_id'].max() + 1)
num_movies = int(second_merge['movie_id'].nunique())

#Embeddings
user_embedding = Embedding(input_dim= num_users,output_dim=64)(user_input)
movie_embedding = Embedding(input_dim = num_movies,output_dim=64)(movie_input)

user_vector = Flatten()(user_embedding)
movie_vector = Flatten()(movie_embedding)

concatenated = Concatenate()([user_vector,movie_vector])

first_dense_layer = Dense(128,activation='relu')(concatenated)
second_dense_layer = Dense(64,activation='relu')(first_dense_layer)
third_dense_layer = Dense(32,activation='relu')(second_dense_layer)

output = Dense(1,activation='linear')(third_dense_layer)

neural_network_model = Model(inputs= [user_input,movie_input],outputs=output)

history = neural_network_model.compile(optimizer='Adam',loss='mean_squared_error')

neural_network_model.summary()

user_ids += 0  

second_merge.loc[:,'movie_id'] = second_merge['final_movie_title'].factorize()[0]
movie_ids = second_merge['movie_id'].values


# Ensure all IDs are within valid range
assert user_ids.min() >= 0 and user_ids.max() < num_users, "Invalid user ID detected!"
assert movie_ids.min() >= 0 and movie_ids.max() < num_movies, "Invalid movie ID detected!"

user_ids = second_merge['user_id'].factorize()[0]
movie_ids = second_merge['movie_id'].factorize()[0]


neural_network_model.fit([user_ids,movie_ids], ratings, epochs=20, batch_size=64)

test_loss =neural_network_model.evaluate([user_ids,movie_ids],ratings)
print(f"Test Loss(MSE) : {test_loss}")



predictions = neural_network_model.predict([user_ids,movie_ids])
for i in range(20):
    print(f"Actual : {ratings[i]},Predicted : {predictions[i][0]}")


def recommend_movies_for_user(user_id,num_recommendations=5):
    watched_movie_ids = set(second_merge[second_merge['user_id']== user_id]['final_movie_title'])


    all_movie_ids = set(second_merge['final_movie_title'].unique())

    unseen_movie_ids = list(all_movie_ids - watched_movie_ids)

    if not unseen_movie_ids:
        print("I’m sorry, I don’t have any new movies for you!")
        return []

    movie_features_array = second_merge.groupby('final_movie_title')[['genre','director']].first()
    
    movie_id_mapping = {title: idx for idx, title in enumerate(second_merge['final_movie_title'].unique())}

    unseen_movie_ids_numeric = [movie_id_mapping[title] for title in unseen_movie_ids]

    user_ids_array = np.array([user_id] * len(unseen_movie_ids_numeric)).reshape(-1,1)

    movie_ids_array = np.array(unseen_movie_ids_numeric).reshape(-1,1)
    
    movie_features = movie_features_array.reindex(unseen_movie_ids)[['genre','director']].dropna().values

    

    missing_ids = [movie_id for movie_id in unseen_movie_ids if movie_id not in movie_features_array.index]
    print(f"Missing movie IDs in features: {missing_ids}")

    predictions = neural_network_model.predict([user_ids_array,movie_ids_array])    

    top_movie_ids = sorted(
        zip(unseen_movie_ids,predictions.flatten()),
        key = lambda x: x[1],
        reverse = True
    )[:num_recommendations]
    return[movie_id for movie_id, _ in top_movie_ids],movie_ids_array
    

recommended_movies, movie_ids = recommend_movies_for_user(3)
print(movie_ids)


user_id = 238
recommended_movies = recommend_movies_for_user(user_id)
print(f"The five recommended movies for user {user_id} are : {recommended_movies}")     
