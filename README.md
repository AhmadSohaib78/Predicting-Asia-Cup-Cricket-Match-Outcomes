# Predicting-Asia-Cup-Cricket-Match-Outcomes
Developed a machine learning model to predict match outcomes in the Asia Cup cricket tournament (1984-2019). Leveraged historical match data and various algorithms including Decision Trees, Logistic Regression, and Support Vector Machines.
# Predicting Asia Cup Cricket Match Outcomes

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Description](#data-description)
- [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
  - [Loading the Dataset](#loading-the-dataset)
  - [Data Exploration](#data-exploration)
  - [Target Class Transformation](#target-class-transformation)
  - [Encoding](#encoding)
- [Data Visualization](#data-visualization)
- [Feature Selection](#feature-selection)
- [Model Building and Evaluation](#model-building-and-evaluation)
  - [Decision Tree](#decision-tree)
  - [Logistic Regression](#logistic-regression)
  - [Naive Bayes](#naive-bayes)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Deep Learning Model](#deep-learning-model)
- [Interpretability Analysis](#interpretability-analysis)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Feature Selection using Wrapper Methods](#feature-selection-using-wrapper-methods)
- [Conclusion](#conclusion)
## Introduction
This project is aimed at predicting match outcomes in the Asia Cup cricket tournament from 1984 to 2019. Cricket enthusiasts, analysts, and stakeholders can leverage these predictions to enhance their understanding and decision-making processes. By applying various machine learning techniques and conducting a thorough analysis of historical match data, this project seeks to develop a predictive model with high accuracy and reliability.

## Project Structure
The project is organized as follows:

- `data/`: Contains the dataset used for the project.
- `notebooks/`: Jupyter notebooks containing the data exploration, preprocessing, model training, and evaluation.
- `models/`: Saved models after training.
- `scripts/`: Python scripts for data processing, feature selection, model training, and evaluation.
- `results/`: Output files such as visualizations, model performance metrics, and SHAP explanations.
- `README.md`: This file, providing an overview and detailed description of the project.

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/predicting-asia-cup-match-outcomes.git
    cd predicting-asia-cup-match-outcomes
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter notebooks or Python scripts:
    ```bash
    jupyter notebook
    ```

## Data Description
The dataset used in this project comprises historical match data from the Asia Cup cricket tournament, covering the years 1984 to 2019. Key features include:

- `team`: Name of the team.
- `opponent`: Name of the opposing team.
- `match_format`: Format of the match (e.g., ODI, T20).
- `ground`: Location where the match was played.
- `toss_outcome`: Result of the toss (e.g., win, lose).
- `batting_statistics`: Batting performance metrics.
- `bowling_statistics`: Bowling performance metrics.
- `result`: Outcome of the match (win, lose, draw).

## Data Preparation and Preprocessing

### Loading the Dataset
The dataset was loaded using the pandas library, ensuring that all relevant features were available for analysis and model building.

### Data Exploration
Comprehensive data exploration was performed to understand the structure, distribution, and characteristics of the dataset. This included checking data types, identifying missing values, and examining the distribution of features.

### Target Class Transformation
The `result` column, which originally contained three classes (win, lose, draw), was transformed to a binary outcome (win or lose) to simplify the classification task and improve model performance.

### Encoding
Label encoding was applied to the `result` column, while one-hot encoding was used for categorical features such as `team`, `opponent`, and `ground`.

## Data Visualization
Data visualization was employed to uncover patterns and relationships within the dataset. Techniques used included:

- Box plots to examine the distribution of numerical features.
- Histograms to analyze the frequency distribution of features.
- Scatter plots to explore relationships between different variables.
- Correlation matrix to identify correlations between features.

## Feature Selection
Feature selection is crucial for building an efficient and interpretable model. Wrapper methods were used to select the most relevant features, optimizing model performance and reducing overfitting.

## Model Building and Evaluation
Several machine learning models were explored to identify the best approach for predicting match outcomes:

### Decision Tree
Decision trees were used for their simplicity and interpretability, allowing us to understand the decision-making process behind match predictions.

### Logistic Regression
Logistic regression served as a baseline model due to its effectiveness in binary classification tasks. It provided a benchmark for comparing more complex models.

### Naive Bayes
Naive Bayes was chosen for its efficiency and performance with high-dimensional data, making it a suitable choice for this task.

### Support Vector Machine (SVM)
SVM was included for its effectiveness in high-dimensional spaces and versatility in classification tasks. It was particularly useful for handling complex patterns in the data.

### Deep Learning Model
In addition to traditional machine learning models, an Artificial Neural Network (ANN) was implemented to leverage deep learning techniques, aiming to enhance prediction accuracy.

## Interpretability Analysis
To gain insights into the logistic regression model's output, the SHAP (SHapley Additive exPlanations) library was used. SHAP values provided a clear understanding of feature importance and model interpretability.

## Hyperparameter Tuning
Hyperparameter tuning was conducted for several models, including Ridge Regression, Linear Regression, Gradient Boosting, Decision Tree, and Naive Bayes. This step was crucial for optimizing model performance and ensuring robustness.

## Feature Selection using Wrapper Methods
Recursive Feature Elimination (RFE), a wrapper method, was employed to iteratively train models and remove the least significant features. This process continued until the desired number of features was reached, resulting in a more focused and effective model.

## Conclusion
This project successfully developed a predictive model for forecasting match outcomes in the Asia Cup cricket tournament. Through comprehensive data preprocessing, feature selection, model building, and evaluation, the project achieved promising results that can be valuable for cricket enthusiasts, stakeholders, and decision-makers.
