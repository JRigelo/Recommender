A recommender system for movies/TV and video games. The recommender takes the
input from the users preferences for some movies and/or games, and suggests the
most suitable movies and video games next.
Code written in Python.

# Dependencies:
 - python 2.7
 - pandas
 - numpy
 - graphlab: http://select.cs.cmu.edu/code/graphlab/

To run this code:

cd src/

 1. Train the models: ipython training_model.py
 1. To get a recommendation: ipython recommender.py

If you want to use the API related to this project go to:
       https://github.com/JRigelo/movies_games_api


# Data:

Datasets contain product ratings from Amazon spanning May 1996 - July 2014
       source: http://jmcauley.ucsd.edu/data/amazon/ (Julian McAuley, UCSD) 

   To acquire the specific datasets used in this model please follow the
   instructions in the source above.
   I used three datasets in this model: movies, video games and a third
   dataset I created containing the common users from movies and games.

# Model Approaches:

  - User-based: users who share the same rating patterns
  - Item-based: relationships between two or more items
  - Matrix factorization (latent factor models): characterizes both items and
  users by vectors  of factors inferred from item rating patterns.

# Implementation:

1. Pre-step: train the model
1. Get feedback from user
1. Find a similar user in the dataset
1. Get first set of recommended items
1. Find similar items on the other two datasets
