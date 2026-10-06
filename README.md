# Regression — Machine Learning

A practical implementation and study repository covering **Regression algorithms in Machine Learning**, with emphasis on understanding the underlying concepts, mathematics, implementation, evaluation, and practical model selection.

This repository is part of my **Machine Learning Mastery journey**, where the goal is to understand *why* a model works, not simply how to call it using Scikit-learn.

---

## 📌 Topics Covered

### 1. Linear Regression

- Simple Linear Regression
- Multiple Linear Regression
- Regression equation
- Coefficients and intercept
- Least Squares Method
- Cost Function
- Residuals
- Prediction
- Model fitting
- Interpretation of coefficients

### 2. Polynomial Regression

- Non-linear relationships
- Polynomial feature transformation
- Degree selection
- Underfitting vs Overfitting
- Model complexity

### 3. Ridge Regression

- L2 Regularization
- Regularization strength
- Controlling model complexity
- Reducing overfitting
- Effect on coefficients

### 4. Lasso Regression

- L1 Regularization
- Feature coefficient shrinkage
- Automatic feature selection
- Sparsity
- Regularization strength

### 5. ElasticNet Regression

- Combination of L1 and L2 regularization
- Feature selection
- Handling correlated features
- Regularization parameters

### 6. SGD Regression

- Stochastic Gradient Descent
- Iterative optimization
- Learning rate
- Mini-batch / stochastic learning concepts
- Large dataset considerations

---

## 🧠 What I Learned

The main objective of this phase was not just to implement regression algorithms, but to understand the reasoning behind them.

### Regression Fundamentals

- What regression problems are
- Continuous target variables
- Relationship between features and target
- Predictions and residuals
- Model parameters
- Cost functions

### Bias–Variance Tradeoff

Understanding the relationship between:

- Underfitting
- Good generalization
- Overfitting
- Model complexity

### Regularization

Understanding why regularization is required and how it affects model parameters.

| Model | Regularization |
|---|---|
| Linear Regression | None |
| Ridge | L2 |
| Lasso | L1 |
| ElasticNet | L1 + L2 |

### Model Evaluation

Regression models were evaluated using metrics such as:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

I also learned why a model should **not be selected using a single metric blindly**.

---

## 📊 Important Regression Metrics

### MAE

Measures the average absolute difference between actual and predicted values.

**Advantages:**

- Easy to interpret
- Less sensitive to outliers than MSE

---

### MSE

Penalizes larger errors more strongly.

**Advantages:**

- Useful when large errors should be heavily penalized
- Differentiable and useful for optimization

---

### RMSE

The square root of MSE.

It is expressed in the **same unit as the target variable**, making it easier to interpret.

---

### R² Score

Measures how much of the variance in the target variable is explained by the model.

A higher R² generally indicates better explanatory performance, but it should not be used alone to judge model quality.

---

## 🔬 Model Comparison

| Model | Main Idea | Regularization | Feature Selection | Non-linear Relationships |
|---|---|---|---|---|
| Linear Regression | Fit a linear relationship | ❌ | ❌ | ❌ |
| Polynomial Regression | Transform features to model curves | ❌ | ❌ | ✅ |
| Ridge | Penalize large coefficients | L2 | ❌ | ❌* |
| Lasso | Shrink coefficients toward zero | L1 | ✅ | ❌* |
| ElasticNet | Combine L1 + L2 | L1 + L2 | ✅ | ❌* |
| SGD Regression | Iterative gradient-based optimization | Optional | ❌ | ❌* |

\*Unless combined with suitable feature transformations.

---

## ⚙️ General Machine Learning Workflow

The regression projects follow a practical workflow:

```text
Problem Definition
       ↓
Understand Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Quality Checks
       ↓
Train/Test Split
       ↓
Feature Preparation
       ↓
Baseline Model
       ↓
Train Regression Models
       ↓
Evaluate Models
       ↓
Compare Models
       ↓
Analyze Errors
       ↓
Select Final Model
       ↓
Interpret Results
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📁 Repository Structure

```text
Regression-/
│
├── README.md
│
├── Linear Regression/
│   └── ...
│
├── Polynomial Regression/
│   └── ...
│
├── Ridge Regression/
│   └── ...
│
├── Lasso Regression/
│   └── ...
│
├── ElasticNet Regression/
│   └── ...
│
└── SGD Regression/
    └── ...
```

> The exact folder structure may evolve as the repository is expanded.

---

## 🎯 Practical Concepts Emphasized

Throughout this phase, special attention was given to:

- Generalization
- Bias vs Variance
- Underfitting vs Overfitting
- Data leakage
- Train/Test splitting
- Feature engineering
- Feature scaling
- Model evaluation
- Residual analysis
- Regularization
- Hyperparameter selection
- Model comparison
- Error analysis

---

## 🔍 Key Takeaways

### Linear Regression

A strong baseline for problems where the relationship between features and target can reasonably be approximated as linear.

### Polynomial Regression

Useful when the relationship is non-linear but can still be represented through polynomial transformations.

### Ridge

Useful when many features contribute to the prediction and coefficient shrinkage can improve generalization.

### Lasso

Useful when feature selection and sparse models are desirable.

### ElasticNet

Useful when combining the benefits of L1 and L2 regularization, particularly when features may be correlated.

### SGD Regression

Useful when iterative optimization is preferable, particularly for larger datasets where traditional approaches may become computationally expensive.

---

## 🚨 Common Mistakes to Avoid

This phase also emphasizes avoiding common regression mistakes:

- Evaluating the model only on training data
- Using the test set repeatedly during development
- Ignoring data leakage
- Choosing a model only because it has the highest R²
- Ignoring outliers
- Using polynomial features with unnecessarily high degrees
- Applying regularization without understanding its effect
- Forgetting feature scaling when required
- Ignoring residual patterns
- Assuming a more complex model is automatically better

---

## 📈 What Comes Next

Regression is one part of my broader **Machine Learning Mastery roadmap**.

### Completed

- 🟢 Linear Regression
- 🟢 Multiple Linear Regression
- 🟢 Polynomial Regression
- 🟢 Ridge Regression
- 🟢 Lasso Regression
- 🟢 ElasticNet Regression
- 🟢 SGD Regression

### Next

**Classification**

The next phase will cover:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Trees
- Random Forest
- Extra Trees
- Support Vector Machines
- Perceptron
- SGD Classifier
- LDA / QDA

---

## 🎓 Learning Philosophy

The goal of this repository is **not to memorize machine learning algorithms**.

The goal is to develop the ability to take an unfamiliar dataset and reason through:

```text
What is the problem?
        ↓
What does the data represent?
        ↓
What should I predict?
        ↓
What preprocessing is required?
        ↓
What should my baseline be?
        ↓
Which models are appropriate?
        ↓
How should I evaluate them?
        ↓
Why did one model perform better?
        ↓
Is the model actually generalizing?
        ↓
What errors is the model making?
        ↓
Which model should I deploy and why?
```

The ultimate objective is to become capable of solving **unfamiliar real-world Machine Learning problems independently**.

---

## 👨‍💻 Author

**Sanjay Akash**

B.Tech — Artificial Intelligence & Data Science

GitHub: [SanjayAkash67](https://github.com/SanjayAkash67)

---

⭐ This repository documents my progression toward practical and independent Machine Learning problem-solving.