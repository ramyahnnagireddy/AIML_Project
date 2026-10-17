# W2D4: Train/Test Split & Cross-Validation

## Objective

Demonstrate train/test splitting and cross-validation techniques using
scikit-learn.

## Topics Covered

- Train/test split
- Stratified train/test split
- K-Fold cross-validation
- Stratified K-Fold cross-validation
- Logistic Regression model evaluation
- Reproducible experiments using fixed random states

## Dataset

The Iris dataset from scikit-learn is used.

- Samples: 150
- Features: 4
- Classes: 3

## Train/Test Split

The dataset is divided into:

- Training samples: 120
- Testing samples: 30
- Test size: 20%

A stratified split is also demonstrated to preserve class proportions.

## Cross-Validation

Five-fold K-Fold and Stratified K-Fold cross-validation are performed.

### K-Fold Results

- Mean accuracy: 0.9733
- Standard deviation: 0.0249

### Stratified K-Fold Results

- Mean accuracy: 0.9667
- Standard deviation: 0.0298

## Testing

The project includes four pytest tests covering:

1. Train/test split sizes
2. Stratified class distribution
3. K-Fold cross-validation
4. Stratified K-Fold cross-validation

All tests passed successfully.

```text
4 passed
```
