# 🚗 Insurance Claims Prediction (ML Pipeline)

End-to-end Machine Learning pipeline for predicting personal auto insurance claim frequency.  
This project focuses on **risk-based pricing**, combining statistical rigor (VIF) with modern ML techniques (XGBoost, pipelines, hyperparameter tuning).

---

## 📌 1. Project Overview

**Goal:**  
Predict whether a policyholder will file a claim, enabling data-driven pricing decisions.

**Approach:**
- Benchmarked 10 machine learning models  
- Used 10-Fold Cross-Validation for robustness  
- Focused on model stability + real-world pricing applicability  

---

## 🧹 2. Data Cleaning & Feature Engineering

### 🔍 Multicollinearity Handling
- Applied Variance Inflation Factor (VIF)  
- Removed redundant features (e.g., `red_vehicle`)  
- Ensured model interpretability and pricing stability  

---

### ⚙️ Preprocessing Pipeline (Scikit-Learn)

Built a fully automated pipeline:

- **Imputation**
  - KNNImputer → numerical features  
  - SimpleImputer → categorical features  

- **Encoding**
  - OrdinalEncoder → ranked features (e.g., Education)  
  - OneHotEncoder → nominal features (e.g., Occupation)  

- **Scaling**
  - StandardScaler  
  - Ensures fair performance for KNN, SVM  

---

## 🤖 3. Model Benchmarking

- Compared 10 classifiers  
- Used 10-Fold Cross-Validation  

📊 *Insert Boxplot Here*  
*CatBoost and XGBoost show highest median accuracy with low variance → strong and stable models.*

---

## ⚡ 4. Hyperparameter Optimization

- **Model:** XGBoost  
- **Method:** RandomizedSearchCV (2000 iterations)  
- **Metric:** Weighted F1-Score  

### Why F1-Score?

In insurance:
- False negatives → underpricing risky drivers  
- False positives → overpricing safe drivers  

👉 F1 ensures a balanced trade-off between precision and recall.

---

## 📊 5. Key Insights (Business Interpretation)

### 🔴 Risk Drivers (High Risk Signals)
- `5_year_num_of_claims` (0.22)  
- `license_points` (0.22)  

👉 Strongest predictors  
👉 Past behavior predicts future risk  

- `single_parent` (0.15)  
👉 Correlates with higher claim frequency  

---

### 🔵 Stability Factors (Protective)
- `value_of_home` (-0.19)  
- `income` (-0.15)  

👉 Financial stability → lower risk  

- `married` (-0.13)  
👉 Statistically safer drivers  

---

### ⚪ Noise Feature
- `red_vehicle` (-0.007)  

👉 No predictive power  
👉 Confirms “red cars are riskier” is a myth  

---

## 🧠 Key Takeaways

- Combining VIF + ML improves reliability  
- XGBoost gives best balance of performance + interpretability  
- Insights align with real-world insurance pricing  

---

## 🚀 Tech Stack

- Python  
- Scikit-Learn  
- XGBoost  
- Pandas / NumPy  
- Matplotlib / Seaborn  

---

## 🔮 Future Improvements

- Add SHAP explainability  
- Build real-time pricing API  
- Explore dynamic pricing (reinforcement learning)  
