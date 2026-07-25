# Smart Agile Task Manager AI Architecture

This document explains the machine learning pipeline, data flow, and technologies used to build the Smart Agile Task Manager AI estimation service.

## 1. Machine Learning Pipeline Overview

The system uses a **Chained Model Architecture** containing two sequential regression models. Instead of predicting Story Points and Actual Hours independently, the architecture mimics how real-world Agile teams estimate effort:

1. **Phase 1 (Complexity Estimation)**: The text and categorical features of a task are passed into a model to predict its complexity (`Story Points`).
2. **Phase 2 (Time Estimation)**: The predicted `Story Points` are injected as an additional feature, alongside the original text and categorical features, into a second model to predict the `Actual Hours`.

By chaining the models, the `Actual Hours` prediction becomes heavily correlated with the `Story Points`, resulting in much more consistent and realistic estimations.

## 2. Feature Engineering

The models cannot read raw text, so the incoming task data must be mathematically transformed into a "Feature Matrix" before training or prediction:

* **Natural Language Processing (NLP)**: The task `Title` and `Description` are combined and converted into numerical vectors using **TF-IDF Vectorization**. This calculates the frequency and importance of specific words (e.g., "authentication", "bug", "database") across all tasks.
* **Ordinal Encoding**: The `Priority` field (Low, Medium, High, Critical) has an inherent mathematical order, so it is converted into integers (`0, 1, 2, 3`).
* **One-Hot Encoding**: The `Task Type` field (Bug, Feature, Chore) has no inherent order, so it is split into separate binary columns (e.g., `Is_Bug: 1`, `Is_Feature: 0`).
* **Matrix Stacking**: Because TF-IDF creates a massive, mostly empty matrix (thousands of columns of zeroes), all features are horizontally stacked into a highly compressed `Scipy Sparse Matrix` to save memory before being fed into the models.

## 3. The Models & Algorithms

The system uses **XGBoost (Extreme Gradient Boosting)** for both prediction models. XGBoost builds a sequence of decision trees, where each new tree specifically tries to correct the errors made by the previous trees.

### Target Variable Normalization
The `Actual Hours` dataset is heavily right-skewed (most tasks take 2-10 hours, but a few take 80+ hours). This skewness heavily degraded model performance. To fix this:
1. During training, a **Logarithmic Transformation** (`np.log1p`) is applied to "Actual Hours" to smooth out extreme outliers and create a normal bell-curve distribution.
2. During prediction, an **Inverse Exponential Transformation** (`np.expm1`) is applied to convert the model's logarithmic output back into true hours.

### Presentation Heuristics
Because Agile teams require strict Fibonacci values (1, 2, 3, 5, 8, 13, 21), the raw float prediction (e.g., `9.24`) is intercepted by Python code and "snapped" to the nearest neighbor (`8`). This snapped integer is then used to generate a rule-based `Complexity` string and `Confidence Score` percentage for the final API response.

## 4. Primary Python Modules Used

The system heavily relies on the following open-source data science ecosystem:

* **`pandas`**: Used for reading the original dataset (Excel/CSV), basic data manipulation, and extracting target variables.
* **`numpy`**: Used for mathematical transformations, specifically applying the log and exponential transformations for the Hours model.
* **`scikit-learn`**: The backbone of the feature engineering pipeline. Provides the `TfidfVectorizer`, `OrdinalEncoder`, and `OneHotEncoder`, as well as standard evaluation metrics (`mean_absolute_error`, `r2_score`).
* **`xgboost`**: Provides the `XGBRegressor` algorithm used for the core machine learning models.
* **`scipy`**: Provides the `csr_matrix` and `hstack` functions used to efficiently manage and merge sparse feature matrices in memory.
* **`joblib`**: Used to serialize (save) and deserialize (load) the trained models and encoders from the hard drive as `.pkl` files.
* **`fastapi` & `uvicorn`**: Used to wrap the prediction logic into an asynchronous, production-ready REST API with automatic Swagger UI documentation.
* **`pydantic`**: Used within FastAPI to strictly validate incoming JSON requests ensuring the API never crashes from malformed data.
