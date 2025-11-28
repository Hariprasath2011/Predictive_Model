# Predictive_Model

1. Problem Statement
The goal of the project is to predict total construction project time and cost using structured project‑level inputs.
This is treated as a multi‑output regression problem.

Inputs include:

    Land size

    Material types + cost

    Number of laborers

    Labor efficiency

    Terrain type

    Weather impact

    Material shortage risk

    Demand–supply factor

    Project type

Outputs:

    Predicted total construction time

    Predicted total construction cost

2. Preprocessing Pipeline:

       Implemented using scikit‑learn ColumnTransformer:

        Numerical features

        Standardization

        Outlier smoothing

       Categorical features

       One‑hot encoding

        Rare-category consolidation

        Risk & environmental features

        Normalization

        Binning for stability

       This preprocessing pipeline is stored as: preprocessor.pkl
3. Machine Learning Approaches Used:
   
       Your model uses three regression algorithms, trained independently:

        (a) Linear Regression
   
       Acts as the baseline model
       Helps understand fundamental relationships
        Useful for explainability

        (b) Random Forest Regressor
   
        Captures nonlinear interactions
        Robust to noise
        Handles high‑dimensional feature expansions from one‑hot encoding

        (c) LightGBM Regressor
   
        Fast, gradient‑boosted decision trees
        Performs extremely well on tabular data
        Supports leaf‑wise growth for high accuracy
        Handles categorical splits efficiently
        These three models are compared based on:RMSE,MAE,R² Score
        The best performing model is saved as:best_model.pkl
   
5. Multi‑Output Architecture:

       Since your prediction has two outputs, you used:

        ✔ MultiOutputRegressor for Linear, RF, and LightGBM
                                OR
        ✔ LightGBM’s native multi‑output handling (if enabled)

          This ensures:

          Joint learning from shared features
        Stable prediction across both cost & time
        Reduced error propagation

