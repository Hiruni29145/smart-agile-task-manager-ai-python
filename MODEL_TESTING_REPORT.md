# Smart Agile Task Manager AI - Testing Report

## 1. Executive Summary
The Smart Agile Task Manager AI was successfully evaluated against a holdout test dataset of 1,000 tasks. The model performs exceptionally well at categorizing task complexity (Story Points) using an XGBoost regression model, achieving an **R² score of 0.746**. The chained Actual Hours model achieved an **R² score of 0.599**. Based on the confusion matrix analysis, the AI rarely makes catastrophic errors and correctly categorizes the majority of tasks, making it a highly reliable tool for Agile estimation.

## 2. Evaluation Methodology
* **Dataset Size:** 5,000 total tasks.
* **Split:** 80% Training (4,000 samples) / 20% Testing (1,000 samples).
* **Architecture:** Chained XGBoost Models. The first model predicts Story Points. The predicted points are injected into a second model to estimate Actual Hours (using a logarithmic transformation to handle skewed data).

## 3. Quantitative Results (Regression Metrics)
The regression models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-Squared (R²). The final chosen model is **XGBoost (XGB)**.

### Story Point Model (Complexity)
* **MAE:** 2.149 (On average, the model's guess is within ~2.1 points of reality).
* **RMSE:** 3.355
* **R²:** 0.746 (Explains 74.6% of the variance, which is excellent for NLP-based complexity).

### Actual Hours Model (Time)
* **MAE:** 7.458 (On average, the model's time guess is within 7.5 hours of actual logged time).
* **RMSE:** 12.272
* **R²:** 0.599 (Explains nearly 60% of variance, representing a solid baseline given that time tracking can be highly subjective).

## 4. Categorical Accuracy & Confusion Matrix
Because Agile estimations rely on strict categories rather than pure decimals, the regression outputs were evaluated after being "snapped" into Complexity Categories: Low, Medium, High, and Very High.

![Confusion Matrix](./confusion_matrix.png)

* **High Accuracy:** The dark diagonal line proves the model is highly accurate at identifying the correct complexity tier.
* **Low Catastrophic Errors:** When the AI does make an error, it is almost exclusively off by only a single adjacent tier. It almost never mistakes a 'Low' complexity task for a 'Very High' complexity task.

## 5. Qualitative Analysis (Sample Predictions - XGBoost)

### Story Points (Complexity)
| True Points | Predicted (Raw) | Difference |
| :--- | :--- | :--- |
| 21 | 21.04 | +0.04 |
| 13 | 13.90 | +0.90 |
| 5 | 4.83 | -0.17 |
| 3 | 2.96 | -0.04 |
| 2 | 1.27 | -0.73 |

### Actual Hours (Time)
| True Hours | Predicted | Difference |
| :--- | :--- | :--- |
| 52.50 | 60.63 | +8.13 |
| 24.50 | 32.41 | +7.92 |
| 11.00 | 5.80 | -5.19 |
| 3.75 | 2.87 | -0.88 |
| 1.00 | 0.61 | -0.38 |

## 6. Conclusion and Limitations
The model is ready for usage by Agile teams. The NLP vectorization correctly extracts context, allowing the model to make nuanced guesses. However, the model occasionally struggles to pinpoint exact hour estimates for mid-range tasks (e.g., estimating 5.8 hours for an 11-hour task). This is expected because time logged by developers often includes unrecorded blockers, making pure textual prediction challenging. The recommended usage is to rely heavily on the **Story Point / Complexity** prediction to guide sprint planning.
