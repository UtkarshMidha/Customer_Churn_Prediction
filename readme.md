# Customer Churn Prediction Using Machine Learning

This project aims to predict customer churn for a telecommunications company using machine learning techniques. By analyzing a customer dataset, we apply data preprocessing, exploratory data analysis (EDA), and machine learning models to predict whether a customer will churn.

## 📝 Project Overview

Customer churn prediction is a critical problem for businesses, as retaining customers is more cost-effective than acquiring new ones. This project leverages various machine learning algorithms such as **Decision Trees**, **Random Forest**, and **XGBoost** to predict customer churn based on demographic and account information.

The workflow covers:

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and addressing class imbalance.
- **Exploratory Data Analysis (EDA)**: Visualizing and analyzing patterns and correlations in the data.
- **Model Training**: Implementing various machine learning algorithms and optimizing them for better performance.
- **Model Evaluation**: Using accuracy, confusion matrix, and classification report to evaluate model performance.
- **Deployment**: Saving the trained model and encoding tools, which can later be used for prediction.

## 📊 Dataset

The dataset used is the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing the following columns:

- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Account Info**: Tenure, MonthlyCharges, TotalCharges
- **Churn**: Target variable indicating whether the customer has churned (Yes/No)

**Dataset Source**: The dataset is publicly available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

## 🛠️ Key Technologies

- **Data Preprocessing**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Model Persistence**: `pickle`
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score

## 🔑 Key Steps

1. **Data Loading**: Loading the raw dataset into a Pandas DataFrame and inspecting its contents.
2. **Data Preprocessing**: Cleaning and preparing the data, including handling missing values, encoding categorical features, and balancing the dataset using **SMOTE**.
3. **Exploratory Data Analysis (EDA)**: Visualizing distributions and relationships between variables to gain insights into the data.
4. **Model Training & Evaluation**: Training various machine learning models, evaluating their performance using cross-validation, and selecting the best model (Random Forest).
5. **Model Deployment**: Saving the trained model and the label encoders using **pickle** for future use.

## 📂 Project Structure

/customer-churn-prediction  
├── data/                        # Dataset file(s)  
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  
│  
├── models/                      # Saved models and encoders  
│   ├── customer_churn_model.pkl  
│   └── encoders.pkl  
│  
├── notebooks/                    # Jupyter Notebook(s) with analysis and modeling  
│   └── main.ipynb  
│  
└── README.md                    # Project documentation


## 📦 How to Use

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   
2. **Install dependencies**:
    ```bash
   pip install -r requirements.txt
   
3. **Run the Jupyter Notebook**:
    ```bash
   jupyter notebook notebook/Customer_Churn_Prediction_using_ML.ipynb
   
## 📊 Model Results

The two models, **Random Forest** and **XGBoost**, were evaluated on the customer churn dataset. The predictions for the same input data showed different results due to the distinct learning mechanisms of both models.

### 🔹 **Random Forest Model**:
- **Prediction**: No Churn
- **Prediction Probability**: `[[0.83 0.17]]` (83% likelihood of **No Churn**)

### 🔹 **XGBoost Model**:
- **Prediction**: Churn
- **Prediction Probability**: `[[0.35321957 0.64678043]]` (64.7% likelihood of **Churn**)

Despite using the same input data, both models arrived at different conclusions because of their inherent differences in the way they learn and make predictions.

---

## 🧐 Conclusion

### 🔑 **Key Insights**:
- **Different Models, Different Results**:  
  - **Random Forest** is a robust model that averages over multiple decision trees, making it less sensitive to noise but sometimes too conservative.
  - **XGBoost** is a gradient boosting model that focuses on correcting previous errors, which makes it more sensitive to specific patterns, often giving a better performance on complex datasets.
  
- **Why the Difference?**:
  - **Random Forest** uses an ensemble of decision trees and makes decisions by majority voting.
  - **XGBoost**, on the other hand, builds trees sequentially, with each tree aiming to correct the previous one, which can lead to more aggressive predictions, especially for minority classes (e.g., **Churn**).

### 📉 **Prediction Differences**:
- While **Random Forest** predicted **No Churn**, **XGBoost** predicted **Churn** with a higher probability. This difference suggests that **XGBoost** might be more sensitive to patterns associated with **Churn**, whereas **Random Forest** might be more balanced.

---

## 🚀 Future Steps

1. **Hyperparameter Tuning** 🔧:
   - Implement **GridSearchCV** or **RandomizedSearchCV** to find the best set of hyperparameters for both models (Random Forest and XGBoost). Hyperparameters like `max_depth`, `n_estimators`, `learning_rate`, and `subsample` can significantly impact model performance.

2. **Model Selection** 🔍:
   - Test additional models such as **Logistic Regression**, **Support Vector Machines**, or **K-Nearest Neighbors**. Evaluate each model's performance using metrics like accuracy, precision, recall, and F1-score to choose the best model for churn prediction.

3. **Downsampling** ⚖️:
   - Try **downsampling** the majority class (No Churn) to balance the dataset, which may improve the model's performance on the minority class (Churn), reducing the impact of class imbalance.

4. **Address Overfitting** 🛠️:
   - Try various techniques to mitigate overfitting:
     - **Pruning** decision trees in Random Forest.
     - Use **early stopping** in XGBoost to stop training once the model performance stops improving.
     - Apply **regularization** techniques in XGBoost like `lambda` and `alpha` parameters.

5. **Stratified K-Fold Cross Validation** 🔄:
   - Implement **Stratified K-Fold Cross Validation** to ensure that each fold has the same proportion of **Churn** and **No Churn** cases, especially when dealing with imbalanced datasets.
   - This will provide a more reliable estimate of model performance across different data splits.

