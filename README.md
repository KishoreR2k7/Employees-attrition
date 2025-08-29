# HR Employee Attrition Prediction using XGBoost

---

## Project Overview
Predict employee attrition (Yes/No) using IBM HR Analytics dataset.  
This model uses **XGBoost Classifier** to identify employees at risk of leaving the organization.

---

## Dataset
- **Source:** [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
- **Number of records:** 1470  
- **Features:** 35 (after cleaning: 32)  
- **Target variable:** `Attrition` (Yes = 1, No = 0)  

**Sample Features:**  
`Age`, `DailyRate`, `DistanceFromHome`, `Education`, `JobLevel`, `JobSatisfaction`, `MonthlyIncome`, `OverTime`, `Department`, `JobRole`, `BusinessTravel`

---

## Data Cleaning & Preprocessing
- Removed constant columns: `EmployeeCount`, `StandardHours`  
- Dropped unnecessary identifiers: `EmployeeNumber`  
- Removed highly correlated feature: `MonthlyIncome`  
- Categorical variables one-hot encoded (e.g., `Department`, `BusinessTravel`, `JobRole`, `OverTime`)  
- Target mapping: `Yes → 1`, `No → 0`  
- Final dataset shape: `1470 rows × 45 columns`  

---

## Exploratory Data Analysis (EDA)
- No missing values or duplicates  
- Class imbalance: 1233 stayed vs 237 left (~1:5 ratio)  
- Key distributions analyzed: `JobInvolvement`, `BusinessTravel`, `OverTime`, `Department`  

---

## Feature Selection
- Removed low-variance features and constant columns  
- Removed highly correlated features to reduce multicollinearity  
- Selected all numeric and one-hot encoded categorical features  

---

## Model
- **Algorithm:** XGBoost Classifier (`xgb.XGBClassifier`)  
- **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation  
```python
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}
Best hyperparameters found:

python
Copy code
{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}
Evaluation Metrics (5-Fold CV)
Metric	Score
Accuracy	0.8701
Precision	0.75
Recall	0.2911
F1-score	0.4195
ROC-AUC	0.8151

Classification Report:

yaml
Copy code
               precision    recall  f1-score   support

           0       0.88      0.98      0.93      1233
           1       0.75      0.29      0.42       237

    accuracy                           0.87      1470
   macro avg       0.81      0.64      0.67      1470
weighted avg       0.86      0.87      0.85      1470
Observations
Model has high accuracy but low recall → misses many employees at risk

Precision is decent → when it predicts leaving, it is often correct

Dataset imbalance affects model performance

Recommendations
Handle class imbalance: SMOTE, ADASYN, or class_weight='balanced'

Adjust prediction threshold to improve recall

Feature engineering to enhance predictive power

Try ensemble models (Random Forest, LightGBM) for comparison

How to Run
Install dependencies:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Load cleaned dataset:

python
Copy code
import pandas as pd
df = pd.read_csv('Cleaned_Data.csv')
Run training and evaluation code provided in hr_attrition.ipynb

File Structure
bash
Copy code
├── Cleaned_Data.csv      # Preprocessed dataset
├── hr_attrition.ipynb    # Notebook with full workflow
├── README.md             # Project description
