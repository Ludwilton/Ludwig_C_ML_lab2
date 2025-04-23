import numpy as np
import pandas as pd
import joblib 
import re
from sklearn.preprocessing import StandardScaler

class RecommendationEngine:
    def __init__(self):
        self.model, self.movie_ids = self._load_recommendation_model() # could use @property
        self.movies_df = pd.read_csv("data/movies.csv")
        self.movie_id_to_title = dict(zip(self.movies_df['movieId'], self.movies_df['title']))
    
    
    def _load_recommendation_model(self):
        """loads the model and movieIDs from the training matrix"""
        model = joblib.load("model_files/nmf_model.joblib")
        movie_ids = joblib.load("model_files/movie_ids.joblib")
        return model, movie_ids
    

    def load_user_ratings(self, path):
        """Load user ratings from CSV and normalize them using StandardScaler"""
        user_data = pd.read_csv(path)
        user_ratings = pd.DataFrame(0.0, index=[0], columns=self.movie_ids)
        
        # making sure we dont add movies that aren't in the model as this causes problems with dimension size
        for _, row in user_data.iterrows():
            movie_id = row["movieId"]
            if movie_id in self.movie_ids:
                user_ratings.loc[0, movie_id] = float(row["Rating"])

        ratings = user_ratings.values[0]
        rated_mask = ratings > 0
        nonzero_ratings = ratings[rated_mask]

        scaler = StandardScaler() # tiny bit more efficient than the notebook approach, identical results
        normalized = scaler.fit_transform(nonzero_ratings.reshape(-1, 1)).flatten()
        normalized = np.clip(normalized, -2.5, 2.5)
        normalized = (normalized + 2.5) / 5.0

        user_ratings.values[0, rated_mask] = normalized
        
        return user_ratings
    
    
    def get_recommendations(self, user_ratings, n_recommendations=100):
        """returns recommendations as a list of movieIDs, sorted from hi-lo model score"""
        user_factors = self.model.transform(user_ratings.values)
        predicted_ratings = user_factors.dot(self.model.components_)[0]

        rated_mask = user_ratings.values[0] > 0
        scores = predicted_ratings.copy()
        scores[rated_mask] = -1
        
        top_indices = np.argsort(-scores)[:n_recommendations]
        recommendations = [self.movie_ids[idx] for idx in top_indices]
        
        return recommendations
    

    def display_recommendations(self, movie_ids):
        """Prints the recommendations as titles"""
        for i, movie_id in enumerate(movie_ids, 1):
            title = self.movie_id_to_title[movie_id]
            print(f"{i}. {title}")


    def find_best_movie_match(self, title): # gpt assisted, mainly because i wanted to spend time on intended functions and fuzzy libs weren't working that well.
        """Primitive seach function to match input string to movies.csv, prioritizing exact matches in format '*title, the* (*year*)'"""
        valid_movies = self.movies_df[self.movies_df['movieId'].isin(self.movie_ids)]
        
        title_lower = title.lower().strip()
        
        exact_matches = valid_movies[valid_movies['title'].str.lower() == title_lower]
        if not exact_matches.empty:
            return exact_matches.iloc[0]
        
        has_year = bool(re.search(r'\(\d{4}\)$', title_lower))
        
        if has_year:
            contains_matches = valid_movies[valid_movies['title'].str.lower().str.contains(
                title_lower, regex=False, na=False)]
            if not contains_matches.empty:
                return contains_matches.iloc[0]
        else:
            for _, movie in valid_movies.iterrows():
                movie_title_lower = movie['title'].lower()
                movie_base = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_title_lower)
                if movie_base.strip() == title_lower:
                    return movie
        
        contains_matches = valid_movies[valid_movies['title'].str.lower().str.contains(
            title_lower, regex=False, na=False)]
        if not contains_matches.empty:
            return contains_matches.iloc[0]
        
        return None
    

    def recommend_from_title(self, title, n_recommendations=10):
        """prints recommendations based on a single string input"""
        match = self.find_best_movie_match(title)
        if match is None:
            print(f"No movies found matching '{title}'")
            return []
        
        movie_id = match["movieId"]
        movie_title = match["title"]
        print(f"Found match for title '{title}' in dataset: {movie_title}")
        
        user_ratings = pd.DataFrame(0.0, index=[0], columns=self.movie_ids)
        
        if movie_id in self.movie_ids:
            user_ratings.loc[0, movie_id] = 1.0
            recommendations = self.get_recommendations(user_ratings, n_recommendations)
            print(f"\nIf you like {movie_title}, you might also enjoy:")
            return recommendations
        else:
            print("Movie not in model database")
            return []
        

    def letterboxd_csv_recommendation(self, csv_path, n_recommendations=100):
        """Load user ratings from CSV and get recommendations"""
        user_ratings = self.load_user_ratings(csv_path)
        recommendations = self.get_recommendations(user_ratings, n_recommendations)
        self.display_recommendations(recommendations)


    def title_recommendation(self, title, n_recommendations=100):
        """Recommends a movie based on a title or a list of titles
        Pass a string in the format "*title* (*YEAR*)"
        
        ** will find similar movies, but not recommended as the model is made to recognize user patterns & the search logic is primitive.
        """
        recommendations = self.recommend_from_title(title, n_recommendations)
        self.display_recommendations(recommendations)

