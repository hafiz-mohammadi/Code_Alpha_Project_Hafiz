**Titanic Survival Prediction**

**Machine Learning Project | CodeAlpha Internship**

This project focuses on predicting passenger survival on the Titanic using supervised machine learning techniques. The work follows a complete data science pipeline, from data exploration to model training and evaluation, using the official Kaggle Titanic dataset.

📌 **Project Overview**

The goal of this project is to analyze passenger data and build classification models that predict whether a passenger survived or not. The project emphasizes data preprocessing, feature engineering, and model comparison.
📂 Dataset
train.csv – training data with survival labels
test.csv – test data without labels
gender_submission.csv – sample submission file
Source: Kaggle Titanic Dataset

🛠️ **Libraries Used**

NumPy – numerical computations
Pandas – data loading, cleaning, and manipulation
Matplotlib & Seaborn – data visualization and exploratory analysis
Scikit-learn –
Preprocessing: SimpleImputer, OneHotEncoder, OrdinalEncoder
Pipelines & transformers: Pipeline, ColumnTransformer
Models: Logistic Regression, Random Forest, SVM, KNN, Decision Tree, Naive Bayes
Model selection & evaluation: train_test_split, cross_val_score, GridSearchCV

🔍 **Exploratory Data Analysis (EDA)**

Analyzed survival rates by gender, passenger class, age, family size, and embarkation port
Visualized age distributions for survived vs non-survived passengers
Identified strong survival patterns (e.g., higher survival for females and first-class passengers)

⚙️ **Feature Engineering**

Created Family Size feature from SibSp and Parch
Grouped family size into categories (Alone, Small, Medium, Large)
Discretized Age into meaningful bins
Encoded categorical features for model compatibility

🤖 **Models Implemented**

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Tree
Gaussian Naive Bayes
Models were trained and evaluated using cross-validation to compare performance.

📊 **Evaluation**
Used accuracy and cross-validation scores
Compared multiple models to identify the best-performing approach
Focused on robustness rather than a single metric

🚀 **Key Learning Outcomes**

Practical experience with real-world data preprocessing
Understanding feature impact on model performance
Hands-on application of multiple ML algorithms
Building clean, reusable ML pipelines
