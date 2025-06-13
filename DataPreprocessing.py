import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# File paths (adjust if needed)
ratings_file = 'u-1.data'
users_file = 'u-1.user'
items_file = 'u-1.item'
genres_file = 'u-1.genre'

# Load ratings file
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_file, sep='\t', names=ratings_cols, engine='python')

# Load users file
users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv(users_file, sep='|', names=users_cols, engine='python')

# Load items file
item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
items = pd.read_csv(items_file, sep='|', names=item_cols, encoding='latin-1', engine='python')

# Load genres file
genre_map = {}
with open(genres_file, 'r') as f:
    for line in f:
        print(line.strip())
        if not line.strip():
            continue    # Skip empty lines
        genre, idx = line.strip().split('|')
        genre_map[int(idx)] = genre

# Optional: Print dataset info
print(f"Loaded {ratings.shape[0]} ratings from {ratings.user_id.nunique()} users and {ratings.item_id.nunique()} movies.")

# --- Split ratings into train and test ---
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# --- Write to train.txt and test.txt ---
train_df.to_csv('train.txt', sep='\t', header=False, index=False)
test_df.to_csv('test.txt', sep='\t', header=False, index=False)

print("✅ Train and test files have been written as 'train.txt' and 'test.txt'.")
