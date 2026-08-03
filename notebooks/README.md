# README Notebooks

Run these in order.

1. `01_pseudo_labeling_feature_engineering.ipynb`
   Builds the severity scoring, temporal decay, and spatial decay logic. Produces the point query scoring function and a feature table used for model training in the next notebook.

2. `02_model_training_baseline.ipynb`
   Trains and compares a baseline, Linear Regression, Random Forest, and Gradient Boosting on the feature table from notebook 1. Saves the winning model and its metadata into `models/` for the API to load.
