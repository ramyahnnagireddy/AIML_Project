This exploratory data analysis examines the Wisconsin diagnostic
dataset using pandas, Matplotlib and Seaborn. The dataset contains
numeric measurements derived from digitized images of breast mass
samples, along with a diagnosis category. Descriptive statistics
summarize the central tendency, spread and range of numeric features.
DataFrame information and missing-value counts help assess data
completeness and column types before further analysis.

Histograms of mean radius, mean texture and mean perimeter show how
these measurements are distributed across diagnosis categories.
Their shapes and overlap provide an initial view of differences
between the groups, but distributions alone cannot establish a
reliable diagnostic rule. A correlation heatmap highlights linear
relationships among ten numeric features selected by variance.
Strong correlations may indicate related measurements or redundant
information, which should be considered in later modelling.

The category-count chart summarizes the available diagnosis labels
and helps reveal whether the classes are balanced. Five observations
are generated from the actual data rather than manually assumed.
This analysis is descriptive, not a clinical assessment or a
validated diagnostic model. Any later predictive modelling would
require careful train-test separation, suitable evaluation metrics,
bias assessment and independent validation. Overall, these outputs
provide a reproducible starting point for understanding the dataset
and planning subsequent machine-learning experiments.
