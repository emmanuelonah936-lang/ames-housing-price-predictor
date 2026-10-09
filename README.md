 Ames Housing Price Prediction
An end-to-end machine learning project that predicts house sale prices using the Ames Housing Dataset. The project covers the full data science workflow — from exploratory data analysis and feature engineering to model tuning, evaluation, and deployment-ready artifacts.

Overview
This project builds a regression model that estimates a home's sale price based on 79 features describing residential home sales in Ames, Iowa (2006–2010). After exploring and cleaning the data, several models were trained and compared, and the best-performing model was selected via cross-validated hyperparameter tuning.

The final model is exported as .pkl files, ready to be served through an interactive web app.

Goals
Demonstrate a complete, end-to-end data science workflow

Handle real-world messy data (missing values, skewed target, mixed feature types)

Compare baseline and advanced regression models

Tune hyperparameters and validate on a held-out test set

Produce deployable artifacts for a user-facing application

Dataset
Source: Ames Housing Dataset — a rich alternative to the classic Boston Housing dataset.

2,930 observations

81 columns (79 predictive features + PID + SalePrice)

Target variable: SalePrice (right-skewed, range $12,789 – $755,000)

Project Workflow
1. Exploratory Data Analysis (EDA)
Structural diagnostics (shape, dtypes, missing values, duplicates)

Target analysis: distribution, skewness (1.74), kurtosis, and outliers

Correlation analysis with SalePrice

Heatmap of the top 15 correlated features

Feature Engineering & Cleaning
Dropped low-variance columns (no predictive value)

Removed data-leakage columns (Mo Sold, Yr Sold, Sale Type, Sale Condition)

Filled structural NaNs ("None" for missing categories, 0 for missing numeric features)

Built a ColumnTransformer pipeline:

Numeric → median imputation + StandardScaler

Categorical → most-frequent imputation + OneHotEncoder

3. Target Transformation
Applied log1p on SalePrice to correct the right skew

Skewness dropped from 1.74 → -0.09 ✅

4. Model Training & Comparison
Model	RMSE	R²
Linear Regression	$31,400	0.8770
Random Forest	$26,815	0.9103
XGBoost	$25,311	0.9201
LightGBM	$24,798	0.9233
Gradient Boosting ⭐	$23,947	0.9285
5. Hyperparameter Tuning
Used GridSearchCV (5-fold CV) on the full pipeline

Best params: learning_rate=0.05, max_depth=4, n_estimators=500

CV R²: 0.8983 ± 0.0095

6. Final Evaluation (Holdout Test Set)
RMSE: $23,835

R²: 0.9291

Residual analysis confirms minimal systematic bias

📊 Key Insights
Top predictive features identified by the final model:

Rank	Feature	Importance
1	Overall Qual	0.491
2	Gr Liv Area	0.121
3	Year Built	0.068
4	Garage Cars	0.037
5	1st Flr SF	0.034
6	Total Bsmt SF	0.027
7	BsmtFin SF 1	0.022
8	Year Remod/Add	0.020
9	Lot Area	0.019
10	Garage Area	0.018
Overall Quality alone accounts for nearly half of the model's predictive power — consistent with real estate domain knowledge.

