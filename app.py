import streamlit as st
import pandas as pd
import os

base_dir = os.path.dirname(__file__)
feedback_file_path = os.path.join(base_dir,'data','feedback.csv')

from Recommendation_Module import recommend_movies_for_user  


st.header("Movie Recommendation System Using AI")
st.title("Movie Recommendation Questionnaire")


user_name = st.text_input("Good day,What's your name?", "")

actors = [
    'Leonardo DiCaprio',
    'Meryl Streep',
    'Denzel Washington',
    'Scarlett Johansson',
    'Robert Downey Jr.',
    'Tom Hanks',
    'Natalie Portman',
    'Morgan Freeman',
    'Brad Pitt',
    'Cate Blanchett',
    'Viola Davis',
    'Will Smith',
    'Emma Stone',
    'Joaquin Phoenix',
    'Jennifer Lawrence',
    'Christian Bale',
    'Ryan Gosling',
    'Charlize Theron',
    'Michael Fassbender',
    'Anne Hathaway'
]

genres = [
    'Fantasy', 'Crime', 'Drama', 'Adventure', 'Action', 
    'Sci-Fi', 'Thriller', 'Documentary', 'Comedy', 
    'Romance', 'Animation', 'Family', 'Musical', 
    'Mystery', 'Western', 'History'
]

directors = [
    'Steven Spielberg', 
    'Christopher Nolan', 
    'Quentin Tarantino', 
    'Martin Scorsese', 
    'Greta Gerwig', 
    'James Cameron', 
    'Gore Verbinski', 
    'Sam Mendes', 
    'Doug Walker', 
    'Andrew Stanton', 
    'Sam Raimi', 
    'Nathan Greno', 
    'Joss Whedon', 
    'David Yates', 
    'Zack Snyder', 
    'Bryan Singer', 
    'Marc Forster', 
    'Andrew Adamson', 
    'David Fincher', 
    'Frank Darabont', 
    'Clint Eastwood'
]

decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
movie_attributes = ['Great plot', 'Character development', 'Action scenes', 'Emotional connection', 'Special effects', 'Dialogue']

fav_actor = st.selectbox("Favourite Actor",actors)
fav_genre = st.selectbox("Favorite Genre", genres)
fav_director = st.selectbox("Favorite Director", directors)
fav_decade = st.selectbox("Favorite Movie Decade", decades)
fav_attributes = st.multiselect("Movie Attributes You Like", movie_attributes)


user_id =user_id = hash(user_name) % 10000  


if st.button("Get Movie Recommendations"):
   
    recommended_movies, movie_ids = recommend_movies_for_user(user_id=user_id, num_recommendations=4)

    
    st.subheader("Recommended Movies:")
    for movie in recommended_movies:
        st.write(movie)  

if 'recommended_movies' in locals() and recommended_movies:
    st.header(f"Thank you for your feedback, {user_name}!")
    feedback = st.slider("Rate the recommendations (1-5 stars):", 1, 5, 3)

    if st.button("Submit Feedback"):
        
        feedback_data = pd.DataFrame({
            'user_name': [user_name],
            'recommended_movies': [recommended_movies],
            'feedback': [feedback],
            'fav_genre': [fav_genre],
            'fav_director': [fav_director],
            'fav_decade': [fav_decade],
            'fav_attributes': [', '.join(fav_attributes)]
        })
        if os.path.exists(feedback_file_path):
            feedback_data.to_csv(feedback_file_path,mode = 'a',header=False,index=False)
        else:
            feedback_data.to_csv(feedback_file_path,mode='w',header=True,index=False)    

        st.write(f"Thank you {user_name}, your feedback has been submitted!")








