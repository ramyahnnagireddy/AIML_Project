# W3D1: Linear Regression - Scikit-Learn

## Objective

Implement and evaluate Linear Regression using Scikit-Learn and compare it with Ridge and Lasso regression.

## Dataset

The Diabetes dataset from Scikit-Learn was used.

- Samples: 442
- Features: 10
- Target: Continuous regression target
- Train samples: 353
- Test samples: 89
- Test size: 20%
- Random state: 42

## Models

Three regression models were trained:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

## Evaluation Metrics

The models were evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

## Results

| Model             |       MSE |    RMSE |     MAE |     R2 |
| ----------------- | --------: | ------: | ------: | -----: |
| Linear Regression | 2900.1936 | 53.8534 | 42.7941 | 0.4526 |
| Ridge             | 3077.4159 | 55.4745 | 46.1389 | 0.4192 |
| Lasso             | 3403.5757 | 58.3402 | 49.7303 | 0.3576 |

For this train/test split, Linear Regression achieved the best results among the three models.

## Linear Regression Coefficients

The trained Linear Regression model produced coefficients for all 10 features:

- age: 37.9040
- sex: -241.9644
- bmi: 542.4288
- bp: 347.7038
- s1: -931.4888
- s2: 518.0623
- s3: 163.4200
- s4: 275.3179
- s5: 736.1989
- s6: 48.6707

## Visualizations

The project generates:

- `predicted_vs_actual.png` - compares actual and predicted values.
- `residuals.png` - displays prediction residuals.

## Output

The model comparison results are also exported to:

`regression_model_comparison.csv`

## Testing

Pytest was used to verify:

- Dataset dimensions
- Train/test split
- Linear Regression training
- Regression metrics
- Ridge and Lasso training
- Evaluation of all three models

Test result:

`6 passed`

## Key Concepts

### Ordinary Least Squares

Linear Regression uses Ordinary Least Squares to find coefficients that minimize the sum of squared prediction errors.

### Ridge Regression

Ridge applies L2 regularization to reduce the impact of large coefficients and help control overfitting.

### Lasso Regression

Lasso applies L1 regularization and can shrink some feature coefficients toward zero.

### Residual

A residual is the difference between an actual value and its predicted value:

`Residual = Actual - Predicted`

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Pytest

## Files

- `linear_regression.py`
- `test_linear_regression.py`
- `regression_model_comparison.csv`
- `predicted_vs_actual.png`
- `residuals.png`
- `README.md`
