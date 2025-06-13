import pandas as pd
import numpy as np

# Load data
users = pd.read_csv('u-1.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
ratings = pd.read_csv('u-1.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
items = pd.read_csv('u-1.item', sep='|', encoding='latin-1', header=None)
movie_titles = items[1]  # Movie title is in column 1

# Load genre mapping
genre_names = []
with open('u-1.genre', 'r') as f:
    genre_names = [line.split('|')[0] for line in f if line.strip()]

# Get unique occupations in the training set
unique_occupations = sorted(users['occupation'].unique())
print("Available occupations:")
for i, occ in enumerate(unique_occupations, 1):
    print(f"{i}. {occ}")

# Prompt user for info
print("\nWelcome to the Movie Recommendation System!")
age = int(input("Enter your age: "))
gender = input("Enter your gender (M/F): ").upper()

# Let user pick occupation from the list
while True:
    try:
        choice = int(input(f"Choose your occupation (1-{len(unique_occupations)}): "))
        if 1 <= choice <= len(unique_occupations):
            occupation = unique_occupations[choice-1]
            break
        else:
            print("Invalid choice. Please try again.")
    except ValueError:
        print("Please enter a number.")

# Find similar users (same gender and occupation, similar age)
similar_users = users[
    (users['gender'] == gender) &
    (users['occupation'] == occupation) &
    (users['age'].between(age-2, age+2))  # +/- 5 years
]

if similar_users.empty:
    print("\nNo similar users found. Showing popular movies.")
    # Calculate average rating for each movie and sort
    popular_movies = ratings.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(10)
    recommendations = popular_movies.index.values
else:
    # Get top-rated movies by similar users
    similar_ratings = ratings[ratings['user_id'].isin(similar_users['user_id'])]
    recommendations = similar_ratings.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(10).index.values

# Recommend movies
print("\nRecommended Movies:")
for i, movie_id in enumerate(recommendations, 1):
    print(f"{i}. {movie_titles[movie_id-1]}")  # Subtract 1 for zero-based index

# Recommend genres (optional: get top genres from recommended movies)
# Genres are in columns 5-23 (0-based index 5:24)
recommended_genres = set()
for movie_id in recommendations:
    movie_genres = items.iloc[movie_id-1, 5:24]  # Subtract 1 for zero-based index
    for i, val in enumerate(movie_genres):
        if val == 1:
            recommended_genres.add(genre_names[i])

print("\nRecommended Genres:")
for genre in sorted(recommended_genres):
    print(f"- {genre}")
