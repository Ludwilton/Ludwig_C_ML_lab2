# Ludwig_C_ML_lab2
backend letterboxd blend movies recommendation for lab showcase



# Model details
Model Algo: Non-negative Matrix Factorization (NMF)
The model was trained on a filtered subset of the MovieLens dataset
User ratings are normalized using z-score standardization
Model performance was evaluated using RMSE metrics, and test user feedback
The reason behind using the collaborative filtering NMF algo was the lack of dimensions in the dataset
NMF finds user patterns and assigns them to a latent feature "n_components"

# Pre-processing Steps
Filtering users by activity level
Selecting diverse users to reduce popularity bias
Filtering movies by popularity
Z-score normalization of user ratings


# Model Performance
The model has been tested with several user profiles, with generally positive feedback. See the demo notebook for examples of recommendations and user feedback.

The search functionality for movie titles is basic and may not find all matches,
mainly because this isnt the intended use but i wanted to keep some easy access testing.
Recommendations may sometimes favor popular movies