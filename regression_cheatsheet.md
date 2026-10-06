""" File formatted for Obsidian Specifically """

# Regression Cheat Sheet

> **Source:** My `Regression-` GitHub repository  
> **Purpose:** Quick syntax reference for future Machine Learning work  
> **Scope:** Syntax actually used in the repository, plus compact reusable patterns derived directly from those implementations.

---

# 1. Core Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Scikit-learn imports used in the repository

```python
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    PoissonRegressor
)

from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
```

---

# 2. Load Dataset

```python
df = pd.read_csv("path/to/file.csv")
```

Examples from the repository:

```python
df = pd.read_csv("datasets/student_performance/student_data.csv")
```

```python
df = pd.read_csv("datasets/ice_cream_sale_temp/IceCreamData.csv")
```

```python
df = pd.read_csv("datasets/housing/housing.csv")
```

---

# 3. Basic Dataset Inspection

```python
df.shape
```

```python
df.head()
```

```python
df.columns
```

```python
df.info()
```

```python
df.describe()
```

---

# 4. Select Features and Target

## Select one feature

```python
X = df[["G2"]]
y = df["G3"]
```

> Double brackets keep `X` as a 2D DataFrame.

## Select multiple features

```python
features = ["studytime", "failures", "absences", "G1", "G2"]

X = df[features]
y = df["G3"]
```

## Select columns directly

```python
X = df[["Temperature"]]
y = df["Revenue"]
```

## Drop target and unwanted columns

```python
X = df.drop(columns=["median_house_value", "ocean_proximity"])
y = df["median_house_value"]
```

---

# 5. Check Shapes

```python
print(X.shape)
print(y.shape)
```

After splitting:

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

Or:

```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

---

# 6. Handle Missing Values

## Drop missing rows from X

```python
X = X.dropna()
```

## Keep y aligned with the remaining X rows

```python
y = y.loc[X.index]
```

## Alternative: drop rows from a selected feature set

```python
X_selected = df[["feature1", "feature2"]].dropna()
y_selected = df.loc[X_selected.index, "target"]
```

---

# 7. Train/Test Split

Standard regression split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Meaning:

- `test_size=0.2` → 20% test data
- `random_state=42` → reproducible split

---

# 8. Simple Linear Regression

## Create model

```python
model = LinearRegression()
```

## Train

```python
model.fit(X_train, y_train)
```

## Predict

```python
y_pred = model.predict(X_test)
```

## Coefficient / slope

For one feature:

```python
print(model.coef_[0])
```

For multiple features:

```python
print(model.coef_)
```

## Intercept

```python
print(model.intercept_)
```

## Score

```python
model.score(X_test, y_test)
```

For `LinearRegression`, `.score()` returns R².

---

# 9. Linear Regression Equation

For one feature:

```text
ŷ = b₀ + b₁x
```

where:

- `b₀` = intercept
- `b₁` = coefficient / slope
- `x` = feature
- `ŷ` = prediction

From the model:

```python
model.intercept_
model.coef_
```

---

# 10. Multiple Linear Regression

```python
features = [
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = df[features]
y = df["G3"]
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Model:

```python
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 11. Polynomial Regression

Polynomial regression in the repository is implemented using `PolynomialFeatures` + `LinearRegression`.

## Create polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

For a higher degree:

```python
poly = PolynomialFeatures(
    degree=10,
    include_bias=False
)
```

## Split transformed features

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)
```

## Train linear regression on polynomial features

```python
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Transform new/unseen data

Important:

```python
X_new_poly = poly.transform(X_new)
```

Do **not** call `fit_transform()` on test/new data.

---

# 12. Polynomial Regression Plot

Sort the original feature for a smooth curve:

```python
sort_index = np.argsort(X["Temperature"])

X_sorted = X.iloc[sort_index]

y_sorted_pred = model.predict(
    poly.transform(X_sorted)
)
```

Plot:

```python
plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_sorted,
    y_sorted_pred,
    linewidth=2,
    label="Polynomial Fit"
)

plt.xlabel("Temperature")
plt.ylabel("Revenue")
plt.title("Polynomial Regression")

plt.legend()
plt.show()
```

---

# 13. Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)
```

Different regularization strength:

```python
Ridge(alpha=0.01)
Ridge(alpha=0.1)
Ridge(alpha=1.0)
Ridge(alpha=10.0)
```

---

# 14. Lasso Regression

```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(
    alpha=0.1,
    max_iter=10000
)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)
```

Inspect coefficients:

```python
lasso_model.coef_
```

---

# 15. ElasticNet Regression

ElasticNet combines L1 and L2 regularization.

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=10000
)

elastic_net.fit(X_train, y_train)

y_pred_en = elastic_net.predict(X_test)
```

Important parameter:

```python
l1_ratio=0
```

→ Ridge-like behavior

```python
l1_ratio=1
```

→ Lasso-like behavior

```python
l1_ratio=0.5
```

→ combination of L1 + L2

---

# 16. SGD Regressor

```python
from sklearn.linear_model import SGDRegressor

sgd_model = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

sgd_model.fit(X_train_scaled, y_train)

y_pred_sgd = sgd_model.predict(X_test_scaled)
```

---

# 17. StandardScaler

Used in the repository for models sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Fit and transform training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Transform test data:

```python
X_test_scaled = scaler.transform(X_test)
```

### Important rule

```python
# Training
scaler.fit_transform(X_train)

# Testing / new data
scaler.transform(X_test)
```

Never fit the scaler separately on the test set.

---

# 18. SVR — Support Vector Regression

Repository implementation:

```python
from sklearn.svm import SVR

svr_model = SVR(
    kernel="rbf",
    C=1000,
    epsilon=0.1
)
```

Train:

```python
svr_model.fit(X_train_scaled, y_train)
```

Predict:

```python
y_pred_svr = svr_model.predict(X_test_scaled)
```

Common kernels:

```python
SVR(kernel="linear")
SVR(kernel="poly")
SVR(kernel="rbf")
```

Repository uses:

```python
SVR(
    kernel="rbf",
    C=1000,
    epsilon=0.1
)
```

---

# 19. Decision Tree Regression

```python
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)
```

Train:

```python
tree_model.fit(X_train, y_train)
```

Predict:

```python
y_pred_tree = tree_model.predict(X_test)
```

Important hyperparameter used:

```python
max_depth=6
```

Other commonly useful parameters:

```python
DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

---

# 20. Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Train:

```python
rf_model.fit(X_train, y_train)
```

Predict:

```python
y_pred_rf = rf_model.predict(X_test)
```

Repository parameters:

```python
n_estimators=100
random_state=42
n_jobs=-1
```

`n_estimators` = number of trees.

`n_jobs=-1` = use available CPU cores.

---

# 21. XGBoost Regression

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
```

Train:

```python
xgb_model.fit(X_train, y_train)
```

Predict:

```python
y_pred_xgb = xgb_model.predict(X_test)
```

Repository parameters:

```python
n_estimators=100
learning_rate=0.1
random_state=42
n_jobs=-1
```

---

# 22. Poisson Regression

Used in the repository to predict a non-negative count target.

```python
from sklearn.linear_model import PoissonRegressor

poisson_model = PoissonRegressor(
    max_iter=500,
    alpha=1e-4
)
```

Train:

```python
poisson_model.fit(X_p_train, y_p_train)
```

Predict:

```python
y_pred_poisson = poisson_model.predict(X_p_test)
```

Example feature selection from the repository:

```python
X_poisson = df[
    [
        "housing_median_age",
        "population",
        "households",
        "median_income"
    ]
].dropna()

y_poisson = df.loc[
    X_poisson.index,
    "total_rooms"
]
```

---

# 23. Regression Metrics

## MAE

```python
mae = mean_absolute_error(y_test, y_pred)
```

## MSE

```python
mse = mean_squared_error(y_test, y_pred)
```

## RMSE

```python
rmse = root_mean_squared_error(y_test, y_pred)
```

## R²

```python
r2 = r2_score(y_test, y_pred)
```

---

# 24. Print Regression Metrics

```python
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
```

Alternative:

```python
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")
```

---

# 25. Compare Actual vs Predicted

```python
comparison = pd.DataFrame({
    "Actual": y_test.values[:5],
    "Predicted": y_pred[:5]
})

print(comparison)
```

---

# 26. Manual MSE from RMSE

```python
mse = rmse ** 2
```

---

# 27. Complete Basic Regression Template

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

# Load data
df = pd.read_csv("data.csv")

# Features and target
X = df[["feature1", "feature2"]]
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
```

---

# 28. Linear Regression with Preprocessing Pipeline

This pattern was used in the repository for mixed numerical + categorical data.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
```

Identify columns:

```python
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns
```

Create preprocessor:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_columns
        )
    ]
)
```

Create pipeline:

```python
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])
```

Train:

```python
model.fit(X_train, y_train)
```

Predict:

```python
y_pred = model.predict(X_test)
```

Evaluate:

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

# 29. OneHotEncoder

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

`handle_unknown="ignore"` prevents errors when test/new data contains an unseen category.

`drop="first"` removes one category to avoid redundant dummy variables in this setup.

---

# 30. ColumnTransformer

Basic pattern:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(), categorical_columns)
    ]
)
```

---

# 31. Pipeline

Basic pattern:

```python
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])
```

Train:

```python
model.fit(X_train, y_train)
```

Predict:

```python
y_pred = model.predict(X_test)
```

---

# 32. Time-Series Regression with Lag Features

The repository contains a lag-based autoregression example.

## Generate dates

```python
dates = pd.date_range(
    start="2025-01-01",
    periods=365,
    freq="D"
)
```

## Generate synthetic trend

```python
trend = np.linspace(50, 150, 365)
```

## Generate seasonality

```python
seasonality = 20 * np.sin(
    np.linspace(0, 3 * np.pi, 365)
)
```

## Generate noise

```python
noise = np.random.normal(0, 5, 365)
```

## Create target

```python
sales = trend + seasonality + noise
```

---

# 33. Create Time-Series DataFrame

```python
ts_df = pd.DataFrame({
    "date": dates,
    "sales": sales
})
```

Set date as index:

```python
ts_df.set_index(
    "date",
    inplace=True
)
```

---

# 34. Create Lag Features

One-period lag:

```python
ts_df["lag_1"] = ts_df["sales"].shift(1)
```

Seven-period lag:

```python
ts_df["lag_7"] = ts_df["sales"].shift(7)
```

Drop missing values:

```python
ts_df = ts_df.dropna()
```

---

# 35. Time-Series X and y

```python
X = ts_df[["lag_1", "lag_7"]]
y = ts_df["sales"]
```

---

# 36. Chronological Train/Test Split

For time-series data, do not randomly shuffle the timeline.

```python
train_size = int(len(ts_df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
```

---

# 37. Time-Series Linear Regression

```python
from sklearn.linear_model import LinearRegression

ts_model = LinearRegression()

ts_model.fit(
    X_train,
    y_train
)

y_pred_ts = ts_model.predict(X_test)
```

Evaluate:

```python
print(
    mean_absolute_error(
        y_test,
        y_pred_ts
    )
)

print(
    root_mean_squared_error(
        y_test,
        y_pred_ts
    )
)

print(
    r2_score(
        y_test,
        y_pred_ts
    )
)
```

---

# 38. Actual vs Predicted Plot

```python
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.index,
    y_test,
    label="Actual Sales"
)

plt.plot(
    y_test.index,
    y_pred_ts,
    label="Predicted Sales",
    linestyle="--"
)

plt.title(
    "Time Series Forecasting with Lag Features"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.show()
```

---

# 39. Scatter Plot

```python
plt.scatter(
    X_test,
    y_test,
    label="Actual Data"
)
```

---

# 40. Line Plot

```python
plt.plot(
    X_test,
    y_pred,
    linewidth=2,
    label="Regression Line"
)
```

---

# 41. Figure Size

```python
plt.figure(figsize=(8, 5))
```

or:

```python
plt.figure(figsize=(12, 5))
```

---

# 42. Labels and Title

```python
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Regression Plot")
```

Legend:

```python
plt.legend()
```

Display:

```python
plt.show()
```

---

# 43. Reproducibility

NumPy:

```python
np.random.seed(42)
```

Scikit-learn:

```python
random_state=42
```

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 44. Model Comparison Pattern

```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "ElasticNet": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=10000
    )
}
```

Train and evaluate:

```python
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(name)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print()
```

---

# 45. Important Model Syntax Summary

| Model | Constructor |
|---|---|
| Linear Regression | `LinearRegression()` |
| Ridge | `Ridge(alpha=0.1)` |
| Lasso | `Lasso(alpha=0.1, max_iter=10000)` |
| ElasticNet | `ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)` |
| SGD Regression | `SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)` |
| Polynomial Regression | `PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression()` |
| SVR | `SVR(kernel="rbf", C=1000, epsilon=0.1)` |
| Decision Tree | `DecisionTreeRegressor(max_depth=6, random_state=42)` |
| Random Forest | `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` |
| XGBoost | `XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)` |
| Poisson | `PoissonRegressor(max_iter=500, alpha=1e-4)` |

---

# 46. Common Workflow to Remember

```text
Load Dataset
     ↓
Inspect Dataset
     ↓
Define X and y
     ↓
Handle Missing Values
     ↓
Train/Test Split
     ↓
Preprocess / Scale if Required
     ↓
Create Model
     ↓
fit()
     ↓
predict()
     ↓
MAE / MSE / RMSE / R²
     ↓
Compare Models
     ↓
Analyze Errors
```

---

# 47. The Three Most Important Syntax Rules

## Rule 1 — Fit only on training data

Correct:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

Incorrect:

```python
scaler.fit_transform(X_test)
```

---

## Rule 2 — Fit model before prediction

```python
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## Rule 3 — Transform new data using an already-fitted transformer

Correct:

```python
poly.fit_transform(X_train)
poly.transform(X_test)
```

Not:

```python
poly.fit_transform(X_test)
```

---

# 48. Regression Metrics Quick Reference

| Metric | Syntax | Lower / Higher |
|---|---|---|
| MAE | `mean_absolute_error(y, pred)` | Lower |
| MSE | `mean_squared_error(y, pred)` | Lower |
| RMSE | `root_mean_squared_error(y, pred)` | Lower |
| R² | `r2_score(y, pred)` | Higher |

---

# 49. Repository Models Covered

The uploaded repository contains these regression implementations:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- SGD Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)
- Poisson Regression
- XGBoost Regression
- Lag-Based Time-Series Regression

---

# 50. Quick Copy-Paste Regression Skeleton

When starting a new regression problem:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

# 1. Load
df = pd.read_csv("data.csv")

# 2. X and y
X = df[["feature1", "feature2"]]
y = df["target"]

# 3. Handle missing values if required
X = X.dropna()
y = y.loc[X.index]

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Model
model = LinearRegression()

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
```

---

# 51. Important Repository Note

In `elastic_sgd.ipynb`, the original repository contains this print label:

```python
print(f"ElasticNet RMSE : {mean_squared_error(y_test, y_pred_en):.4f}")
```

That expression is actually calculating **MSE**, not RMSE.

Correct syntax:

```python
print(
    f"ElasticNet MSE : "
    f"{mean_squared_error(y_test, y_pred_en):.4f}"
)

print(
    f"ElasticNet RMSE : "
    f"{root_mean_squared_error(y_test, y_pred_en):.4f}"
)
```

The same labeling issue appears in the SGD section.

---

# 52. Final Regression Reminder

The important pattern is:

```python
X
y
↓
train_test_split()
↓
preprocessing
↓
model = Model(...)
↓
model.fit()
↓
model.predict()
↓
MAE
MSE
RMSE
R²
↓
error analysis
↓
model comparison
```

Do not choose a regression model only because it gives the highest R².

Consider:

- Generalization
- MAE / RMSE
- Residuals
- Outliers
- Data size
- Feature types
- Non-linearity
- Scaling requirements
- Regularization
- Interpretability
- Computational cost
- Overfitting
- Business requirements

---

## End

This cheat sheet is intended as a **syntax reference**, not a replacement for understanding the regression concepts.
