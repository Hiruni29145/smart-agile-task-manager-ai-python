==================================================
SMART AGILE TASK MANAGER AI
==================================================

Overview
--------
The Smart Agile Task Manager AI is a machine learning-powered estimation tool designed to help Agile teams predict the complexity and time required for software development tasks. 

Features
--------
* Predicts Agile Story Points (snapped to standard Fibonacci sequence values: 1, 2, 3, 5, 8, 13, 21).
* Estimates actual completion hours based on task description and complexity.
* Assigns a heuristic Complexity rating (Low, Medium, High, Very High) and Confidence Score.
* Powered by TF-IDF Natural Language Processing and XGBoost regression models.
* Served via a fast, interactive REST API using FastAPI.

Directory Structure
-------------------
* /models   - Contains the serialized XGBoost .pkl models and TF-IDF vectorizers.
* /src      - Contains the source code for the ML pipeline and FastAPI server.
* /dataset  - Contains the original Excel dataset (if applicable).

Documentation
-------------
Please refer to the following Markdown files for detailed information:
* ARCHITECTURE.md - Explains the ML pipeline, feature engineering, and algorithms used.
* SERVER_GUIDE.md - Provides step-by-step instructions on how to start the API server locally.
