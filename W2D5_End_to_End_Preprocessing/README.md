\# W2D5: End-to-End Preprocessing Pipeline



\## Objective



Apply the Week 2 machine learning preprocessing concepts to the Titanic dataset and produce an ML-ready feature dataset.



\## Dataset



The Titanic dataset contains 891 rows and 15 original columns.



The dataset was loaded using pandas from the public seaborn-data repository.



\## Preprocessing Pipeline



The following steps were performed:



1\. \*\*Exploratory Data Analysis (EDA)\*\*



&#x20;  \* Inspected dataset shape

&#x20;  \* Examined column names and data types

&#x20;  \* Identified missing values

&#x20;  \* Displayed sample records and summary statistics



2\. \*\*Missing Value Handling\*\*



&#x20;  \* Missing `age` values were replaced using the median.

&#x20;  \* Missing `embarked` values were replaced using the mode.

&#x20;  \* Missing `embark\_town` values were replaced using the mode.

&#x20;  \* `deck` was removed because it contained a large proportion of missing values.



3\. \*\*Data Leakage Prevention\*\*



&#x20;  \* The `alive` column was removed because it directly represents the target `survived`.



4\. \*\*Categorical Encoding\*\*



&#x20;  \* One-hot encoding was applied to:



&#x20;    \* `sex`

&#x20;    \* `embarked`

&#x20;    \* `class`

&#x20;    \* `who`

&#x20;    \* `embark\_town`

&#x20;  \* Boolean columns `adult\_male` and `alone` were converted to integer values.



5\. \*\*Feature Scaling\*\*



&#x20;  \* StandardScaler was applied to:



&#x20;    \* `pclass`

&#x20;    \* `age`

&#x20;    \* `sibsp`

&#x20;    \* `parch`

&#x20;    \* `fare`

&#x20;  \* Encoded binary features were kept as 0/1 values.



6\. \*\*Export\*\*



&#x20;  \* The processed features and target variable were exported to:

&#x20;    `titanic\_processed.csv`



\## Final Dataset



| Property                       | Value |

| ------------------------------ | ----: |

| Original rows                  |   891 |

| Original columns               |    15 |

| Processed feature columns      |    21 |

| Final columns including target |    22 |

| Missing values                 |     0 |



\## Files



```text

W2D5\_End\_to\_End\_Preprocessing/

├── titanic\_preprocessing.py

├── test\_titanic\_preprocessing.py

├── titanic\_processed.csv

└── README.md

```



\## Testing



The preprocessing pipeline was validated using pytest.



Tests verify:



\* Processed CSV exists

\* Final dataset shape is correct

\* No missing values remain

\* Target column exists

\* Target contains both classes



\## Result



The Titanic dataset was successfully transformed from raw data into a clean, encoded, scaled, and ML-ready dataset.



