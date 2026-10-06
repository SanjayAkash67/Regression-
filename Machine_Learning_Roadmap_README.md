# Machine Learning Learning Roadmap

## Goal

Learn Machine Learning from fundamentals to advanced practical modeling through projects.

The goal is not to memorize algorithms.

The goal is to understand:
- What a model does
- How the algorithm works
- Why it works
- When to use it
- How to prepare data
- How to train it
- How to evaluate it
- How to improve it
- How to compare models
- How to avoid common ML mistakes
- How to build complete ML systems

---

# Learning Philosophy

```text
ONE TOPIC
    ↓
UNDERSTAND THE CONCEPT
    ↓
UNDERSTAND THE INTUITION
    ↓
LEARN THE IMPORTANT MATHEMATICS
    ↓
IMPLEMENT IT
    ↓
BUILD A PRACTICAL PROJECT
    ↓
EVALUATE THE MODEL
    ↓
UNDERSTAND THE RESULTS
    ↓
DOCUMENT IT
    ↓
MOVE TO THE NEXT TOPIC
```

Machine Learning will be learned primarily through practical projects.

---

# Phase 0 — Prerequisites

These topics can be learned before and alongside Machine Learning.

## Python
- Variables
- Data types
- Conditions
- Loops
- Functions
- Lists, tuples, dictionaries, sets
- Comprehensions
- Modules
- Exceptions
- File handling
- Basic OOP

## NumPy
- Arrays
- Shapes
- Indexing
- Slicing
- Vectorized operations
- Mathematical operations
- Statistics
- Broadcasting
- Linear algebra basics

## Pandas
- DataFrames
- Data loading
- Data cleaning
- Filtering
- Sorting
- GroupBy
- Merge / Join
- Pivot tables
- Time series
- String operations

## Statistics
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Percentiles
- Probability
- Distributions
- Correlation
- Covariance

## Data Visualization
- Matplotlib
- Seaborn
- Histograms
- Box plots
- Scatter plots
- Bar charts
- Line charts
- Heatmaps

> NumPy, Statistics, Data Visualization, Feature Engineering and EDA can continue in parallel with Machine Learning.

---

# Phase 1 — Machine Learning Fundamentals

## 1. What is Machine Learning?
- AI vs ML
- ML vs Deep Learning
- Traditional programming vs ML
- How ML learns from data
- Features
- Target / label
- Samples
- Training data
- Testing data
- Validation data
- Parameters
- Hyperparameters

## 2. Types of Machine Learning

### Supervised Learning
- Regression
- Classification

### Unsupervised Learning
- Clustering
- Dimensionality Reduction
- Anomaly Detection

### Reinforcement Learning
- Basic understanding
- Detailed RL will be studied separately later

## 3. Machine Learning Workflow

```text
Problem Definition
        ↓
Collect Data
        ↓
Understand Data
        ↓
Clean Data
        ↓
EDA
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Preprocessing
        ↓
Model Selection
        ↓
Training
        ↓
Prediction
        ↓
Evaluation
        ↓
Hyperparameter Tuning
        ↓
Model Comparison
        ↓
Final Model
        ↓
Save Model
        ↓
Deployment
```

---

# Phase 2 — Scikit-Learn Fundamentals

## 4. Scikit-Learn
- Scikit-learn ecosystem
- Estimators
- Transformers
- Predictors
- `fit()`
- `predict()`
- `transform()`
- `fit_transform()`
- Train/test splitting
- Random states
- Model APIs

Important tools:
- `train_test_split`
- `Pipeline`
- `ColumnTransformer`

---

# Phase 3 — Data Preprocessing

## 5. Missing Values
- Detecting missing values
- Mean imputation
- Median imputation
- Mode imputation
- Constant-value imputation
- `SimpleImputer`

## 6. Categorical Data
- Nominal variables
- Ordinal variables
- Label encoding
- Ordinal encoding
- One-hot encoding
- Handling unknown categories
- `OneHotEncoder`
- `OrdinalEncoder`

## 7. Feature Scaling
- Why scaling matters
- Standardization
- Min-Max scaling
- Robust scaling
- When scaling is necessary
- When scaling is unnecessary
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

## 8. Feature Transformation
- Log transformation
- Power transformation
- Normalization
- Distribution transformation
- Numerical transformations

## 9. Data Leakage
- What data leakage is
- Why leakage is dangerous
- Leakage during preprocessing
- Leakage during feature engineering
- Leakage during train/test splitting
- How Pipelines help prevent leakage

---

# Phase 4 — Regression

## 10. Linear Regression
### Project: House Price Prediction

Learn:
- Linear relationship
- Equation of a line
- Coefficients
- Intercept
- Least squares
- Residuals
- Predictions

Metrics:
- MAE
- MSE
- RMSE
- R²

## 11. Multiple Linear Regression
### Project: Salary Prediction
- Multiple features
- Coefficients
- Feature contribution
- Multivariable regression

## 12. Polynomial Regression
### Project: Fuel Consumption Prediction
- Non-linear relationships
- Polynomial features
- Polynomial degree
- Underfitting
- Overfitting

## 13. Ridge Regression
### Project: Regularized House Price Prediction
- Regularization
- L2 penalty
- Bias-variance tradeoff
- Alpha

## 14. Lasso Regression
### Project: Feature Selection & Price Prediction
- L1 regularization
- Sparse models
- Automatic feature selection
- Alpha

## 15. ElasticNet Regression
### Project: Mixed Regularization Predictor
- L1 + L2 regularization
- ElasticNet
- Alpha
- L1 ratio

---

# Phase 5 — Classification

## 16. Classification Fundamentals
- Binary classification
- Multiclass classification
- Multilabel classification
- Class labels
- Decision boundaries
- Probabilities
- Classification thresholds

## 17. Logistic Regression
### Project: Diabetes Prediction
- Sigmoid function
- Probability prediction
- Decision threshold
- Log loss
- Binary classification

## 18. K-Nearest Neighbors
### Project: Iris Classifier
- Distance
- Nearest neighbors
- K value
- Decision boundaries
- Feature scaling

## 19. Naive Bayes
### Project: Spam Detector
- Bayes theorem
- Conditional probability
- Naive independence assumption
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

## 20. Decision Trees
### Project: Loan Approval System
- Tree structure
- Nodes
- Splits
- Entropy
- Information gain
- Gini impurity
- Maximum depth
- Overfitting

## 21. Random Forest
### Project: Customer Churn Prediction
- Ensemble learning
- Bagging
- Multiple decision trees
- Random feature selection
- Feature importance

## 22. Extra Trees
### Project: Employee Attrition Prediction
- Extremely randomized trees
- Random Forest vs Extra Trees
- Ensemble variance reduction

## 23. Support Vector Machines
### Project: Cancer Classification
- Hyperplanes
- Margins
- Support vectors
- Kernel trick
- Linear kernel
- RBF kernel
- C
- Gamma
- Feature scaling

---

# Phase 6 — Boosting Algorithms

## 24. Gradient Boosting
### Project: Customer Risk Prediction
- Boosting
- Weak learners
- Sequential learning
- Residual errors
- Learning rate
- Number of estimators
- Tree depth

## 25. AdaBoost
### Project: Fraud Classification
- Adaptive boosting
- Weak learners
- Sample weighting
- Sequential ensemble learning

## 26. XGBoost
### Project: Customer Churn Predictor
- Gradient boosting trees
- Regularization
- Learning rate
- Tree depth
- Subsampling
- Feature importance
- Early stopping

## 27. LightGBM
### Project: Large-Scale Classification
- Gradient boosting
- Histogram-based learning
- Leaf-wise growth
- Large datasets
- Speed and efficiency

## 28. CatBoost
### Project: Categorical Customer Prediction
- Gradient boosting
- Native categorical feature handling
- Ordered boosting
- Categorical-heavy datasets

---

# Phase 7 — Model Evaluation

## 29. Regression Metrics
- MAE
- MSE
- RMSE
- R²
- Adjusted R²
- MAPE
- When to use each metric

## 30. Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Sensitivity
- ROC-AUC
- PR-AUC
- Log loss

## 31. Confusion Matrix

```text
                 Predicted
              Positive  Negative

Actual
Positive         TP        FN

Negative         FP        TN
```

Learn:
- True Positive
- True Negative
- False Positive
- False Negative

## 32. Classification Thresholds
- Probability thresholds
- Precision-recall tradeoff
- Threshold tuning
- Business-dependent thresholds

---

# Phase 8 — Cross Validation

## 33. Cross-Validation
- Why one train/test split can be unreliable
- K-Fold Cross Validation
- Stratified K-Fold
- Regression cross-validation
- Classification cross-validation
- Cross-validation scores
- `cross_val_score`
- `cross_validate`

---

# Phase 9 — Hyperparameter Tuning

## 34. Hyperparameters
- Parameters vs hyperparameters
- Manual tuning
- Grid Search
- Random Search
- Search spaces
- Scoring
- Cross-validation during tuning
- Best parameters
- Best estimator
- `GridSearchCV`
- `RandomizedSearchCV`

---

# Phase 10 — ML Pipelines

## 35. Pipelines

```text
Missing Value Handling
        ↓
Encoding
        ↓
Scaling
        ↓
Feature Engineering
        ↓
Model
```

Learn:
- `Pipeline`
- `ColumnTransformer`
- Reproducibility
- Leakage prevention
- Cleaner code
- Production-ready preprocessing

---

# Phase 11 — Feature Selection

## 36. Feature Selection

### Filter Methods
- Correlation
- Variance threshold
- Statistical tests

### Wrapper Methods
- Recursive Feature Elimination
- RFE

### Embedded Methods
- Lasso
- Tree-based feature importance

Understand:
- Irrelevant features
- Redundant features
- Multicollinearity
- Feature selection
- Fewer features vs model performance

---

# Phase 12 — Clustering

## 37. K-Means Clustering
### Project: Customer Segmentation
- Clusters
- Centroids
- Distance
- Iterative optimization
- K value
- Inertia
- Elbow method
- Silhouette score

## 38. Hierarchical Clustering
### Project: Mall Customer Segmentation
- Agglomerative clustering
- Linkage
- Distance
- Dendrograms
- Number of clusters

## 39. DBSCAN
### Project: Geographic / Location Clustering
- Density-based clustering
- Core points
- Border points
- Noise
- Epsilon
- Minimum samples

## 40. Gaussian Mixture Models
### Project: Customer Behavior Clustering
- Probability-based clustering
- Gaussian distributions
- Mixture models
- Soft clustering

---

# Phase 13 — Dimensionality Reduction

## 41. PCA
### Project: High-Dimensional Dataset Visualization
- Dimensionality reduction
- Variance
- Principal components
- Eigenvectors
- Eigenvalues
- Explained variance
- Feature transformation

## 42. t-SNE
### Project: Embedding Visualization
- Non-linear dimensionality reduction
- Local structure
- High-dimensional visualization
- Perplexity
- Interpretation limitations

## 43. UMAP
### Project: Advanced Dimensionality Reduction
- Manifold learning
- Local structure
- Global structure
- UMAP parameters
- PCA vs t-SNE vs UMAP

---

# Phase 14 — Anomaly Detection

## 44. Isolation Forest
### Project: Transaction Anomaly Detection
- Anomalies
- Isolation
- Random trees
- Contamination

## 45. One-Class SVM
### Project: Network Anomaly Detector
- One-class classification
- Decision boundaries
- Support vectors
- Kernel-based anomaly detection

## 46. Local Outlier Factor
### Project: Fraudulent Transaction Detection
- Local density
- Neighbor comparison
- Local outliers
- Density-based anomaly detection

---

# Phase 15 — Advanced ML Concepts

## 47. Underfitting vs Overfitting
- Bias
- Variance
- Bias-variance tradeoff
- Training error
- Validation error
- Generalization

## 48. Regularization
- L1
- L2
- ElasticNet
- Regularization strength
- Why regularization works

## 49. Feature Importance
- Linear model coefficients
- Tree feature importance
- Permutation importance

## 50. Model Interpretability
- Why models make predictions
- Global interpretation
- Local interpretation
- Feature contribution
- SHAP concepts
- Explainable ML

## 51. Imbalanced Classification
- Class imbalance
- Why accuracy can be misleading
- Precision / recall tradeoff
- Stratified splitting
- Class weights
- Resampling
- SMOTE concepts
- PR-AUC

## 52. Multicollinearity
- Correlated features
- Redundant information
- Variance Inflation Factor
- Effects on linear models
- Feature selection

## 53. Probability Calibration
- Predicted probabilities
- Calibration
- Reliability
- Calibration curves
- Why probability quality matters

---

# Phase 16 — Practical ML Engineering

## 54. Reproducibility
- Random seeds
- Reproducible experiments
- Dataset versions
- Environment management
- Requirements files

## 55. Model Persistence
- Saving trained models
- Loading trained models
- `joblib`
- Model artifacts

## 56. ML Project Structure

```text
project/
│
├── data/
├── notebooks/
├── src/
├── models/
├── output/
├── app.py
├── requirements.txt
└── README.md
```

## 57. Complete ML Pipeline

```text
Dataset
   ↓
Data Understanding
   ↓
Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Model
   ↓
Cross Validation
   ↓
Hyperparameter Tuning
   ↓
Evaluation
   ↓
Model Interpretation
   ↓
Save Model
   ↓
Prediction Application
```

---

# Phase 17 — Model Comparison

Learn how to answer:

- Which model performs better?
- Why does it perform better?
- Which metric matters?
- Is the model overfitting?
- Does the model need scaling?
- Is the dataset imbalanced?
- Are there too many features?
- Is there data leakage?
- Does the model generalize?
- What are the computational costs?
- Is the model interpretable?
- Is the model suitable for deployment?

---

# Phase 18 — End-to-End Machine Learning Projects

## Project 1 — House Price Prediction

Skills:
- Regression
- EDA
- Feature engineering
- Preprocessing
- Pipelines
- Cross-validation
- Hyperparameter tuning
- Model comparison
- Model evaluation

## Project 2 — Customer Churn Prediction

Skills:
- Classification
- Encoding
- Scaling
- Imbalanced data
- Evaluation
- Feature importance
- Model comparison

## Project 3 — Fraud Detection System

Skills:
- Classification
- Imbalanced datasets
- Precision / Recall
- Anomaly detection
- Threshold tuning
- Model evaluation

## Project 4 — Customer Segmentation

Skills:
- EDA
- Scaling
- K-Means
- Hierarchical clustering
- DBSCAN
- PCA
- Visualization

## Project 5 — Complete ML Prediction System

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Preprocessing
 ↓
Multiple Models
 ↓
Cross Validation
 ↓
Hyperparameter Tuning
 ↓
Model Comparison
 ↓
Best Model
 ↓
Interpretation
 ↓
Save Model
 ↓
Prediction API / Application
```

---

# Complete Algorithm Checklist

## Regression
- [ ] Linear Regression
- [ ] Multiple Linear Regression
- [ ] Polynomial Regression
- [ ] Ridge Regression
- [ ] Lasso Regression
- [ ] ElasticNet

## Classification
- [ ] Logistic Regression
- [ ] KNN
- [ ] Naive Bayes
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Extra Trees
- [ ] SVM
- [ ] Gradient Boosting
- [ ] AdaBoost
- [ ] XGBoost
- [ ] LightGBM
- [ ] CatBoost

## Clustering
- [ ] K-Means
- [ ] Hierarchical Clustering
- [ ] DBSCAN
- [ ] Gaussian Mixture Models

## Dimensionality Reduction
- [ ] PCA
- [ ] t-SNE
- [ ] UMAP

## Anomaly Detection
- [ ] Isolation Forest
- [ ] One-Class SVM
- [ ] Local Outlier Factor

---

# Main Python Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Joblib

---

# Machine Learning Skills Checklist

- [ ] Explain Machine Learning
- [ ] Explain supervised learning
- [ ] Explain unsupervised learning
- [ ] Explain regression
- [ ] Explain classification
- [ ] Explain clustering
- [ ] Define features and targets
- [ ] Prepare datasets
- [ ] Split datasets correctly
- [ ] Handle missing values
- [ ] Encode categorical variables
- [ ] Scale numerical features
- [ ] Build preprocessing pipelines
- [ ] Prevent data leakage
- [ ] Build regression models
- [ ] Build classification models
- [ ] Build clustering models
- [ ] Perform dimensionality reduction
- [ ] Detect anomalies
- [ ] Evaluate regression models
- [ ] Evaluate classification models
- [ ] Read confusion matrices
- [ ] Use cross-validation
- [ ] Tune hyperparameters
- [ ] Compare multiple models
- [ ] Handle imbalanced datasets
- [ ] Perform feature selection
- [ ] Understand multicollinearity
- [ ] Interpret model predictions
- [ ] Understand model calibration
- [ ] Save trained models
- [ ] Build complete ML pipelines
- [ ] Build end-to-end ML projects
- [ ] Explain why a particular model was selected
- [ ] Prepare a model for deployment

---

# Definition of Machine Learning Completed

Machine Learning is considered practically complete when I can take an unfamiliar dataset and independently:

```text
Understand the Problem
        ↓
Understand the Dataset
        ↓
Clean the Data
        ↓
Perform EDA
        ↓
Engineer Features
        ↓
Prepare Preprocessing
        ↓
Choose Suitable Models
        ↓
Train Models
        ↓
Cross-Validate
        ↓
Tune Hyperparameters
        ↓
Evaluate Correctly
        ↓
Compare Models
        ↓
Interpret Results
        ↓
Select Final Model
        ↓
Save Model
        ↓
Build a Usable Prediction System
```

---

# Final Goal

The goal is NOT:

> "I know many Machine Learning algorithms."

The goal is:

> "Given a real-world problem and an unfamiliar dataset, I can independently understand the data, choose an appropriate ML approach, prepare the data, build models, evaluate them correctly, improve them, explain the results, select the best model, and turn it into a usable system."

---

# Position in the Bigger AI Roadmap

```text
Pandas
   ↓
NumPy ───────────────────────┐
   ↓                         │
Statistics ──────────────────┤
   ↓                         │
Data Visualization ──────────┤
   ↓                         │
Feature Engineering ─────────┤
   ↓                         │
EDA ─────────────────────────┤
                             │
                             ↓
                    MACHINE LEARNING
                           CURRENT
                             ↓
                    Deep Learning
                             ↓
                    Computer Vision
                             ↓
                          NLP
                             ↓
                    LLM Engineering
                             ↓
                      Time Series
                             ↓
                Reinforcement Learning
                             ↓
                 Recommendation Systems
                             ↓
                     Generative AI
                             ↓
                          MLOps
                             ↓
                  End-to-End AI Systems
```

---

# Learning Rule

For every Machine Learning topic:

```text
1. Understand the problem
2. Understand the algorithm
3. Understand the intuition
4. Learn the important mathematics
5. Learn the implementation
6. Build a practical project
7. Evaluate the model
8. Compare results
9. Understand strengths and weaknesses
10. Move to the next topic
```

Do not try to master every mathematical detail before building projects.

Learn the mathematics deeply enough to understand what the algorithm is doing, then strengthen the mathematics as the projects become more advanced.
