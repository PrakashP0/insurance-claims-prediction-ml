🚗 Insurance Claims Prediction (ML Pipeline)

End-to-end Machine Learning pipeline for predicting personal auto insurance claim frequency.
This project focuses on risk-based pricing, combining statistical rigor (VIF) with modern ML techniques (XGBoost, pipelines, hyperparameter tuning).

📌 1. Project Overview

Goal:
Predict whether a policyholder will file a claim, enabling data-driven pricing decisions.

Approach:

Benchmarked 10 machine learning models
Used 10-Fold Cross-Validation for robustness
Focused on model stability + real-world pricing applicability
🧹 2. Data Cleaning & Feature Engineering
🔍 Multicollinearity Handling
Applied Variance Inflation Factor (VIF)
Removed redundant features (e.g., red_vehicle)
Ensured model interpretability and pricing stability
⚙️ Preprocessing Pipeline (Scikit-Learn)

Built a fully automated pipeline:

Imputation
KNNImputer → numerical features
SimpleImputer → categorical features
Encoding
OrdinalEncoder → ranked features (e.g., Education)
OneHotEncoder → nominal features (e.g., Occupation)
Scaling
StandardScaler
Ensured fair performance for distance-based models (KNN, SVM)
🤖 3. Model Benchmarking
Compared 10 classifiers
Used 10-Fold Cross-Validation

📊 Insert Boxplot Here
Caption: CatBoost and XGBoost show the highest median accuracy with low variance, indicating strong and stable performance.

⚡ 4. Hyperparameter Optimization
Model: XGBoost
Method: RandomizedSearchCV (2000 iterations)
Evaluation Metric: Weighted F1-Score

Why F1-Score?
In insurance:

False negatives → underpricing risky drivers
False positives → overpricing safe drivers

F1 ensures a balanced trade-off between precision and recall.

📊 5. Key Insights (Business Interpretation)
🔴 Frequency Predictors (High Risk Signals)
5_year_num_of_claims (0.22)
license_points (0.22)
👉 Strongest predictors of future claims
👉 Confirms actuarial principle: past behavior predicts future risk
single_parent (0.15)
👉 Correlates with higher claim frequency (contextual lifestyle factors)
🔵 Stability Predictors (Protective Factors)
value_of_home (-0.19)
income (-0.15)
👉 Financial stability → lower risk behavior
married (-0.13)
👉 Statistically safer drivers → lower premiums
⚪ Noise Features
red_vehicle (-0.007)
👉 No predictive power
👉 Confirms: “Red cars are riskier” is a myth
🧠 Key Takeaways
Combining statistical methods (VIF) with ML models improves reliability
XGBoost provides the best balance of performance + interpretability
Feature insights align with real-world insurance pricing logic
🚀 Tech Stack
Python
Scikit-Learn
XGBoost
Pandas / NumPy
Matplotlib / Seaborn
📌 Future Improvements
Add SHAP values for explainability
Integrate real-time pricing API
Explore dynamic pricing models (reinforcement learning)
