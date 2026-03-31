# insurance-claims-prediction-ml
"End-to-end Machine Learning pipeline for predicting personal auto insurance claim frequency. Features VIF-based multicollinearity checks, automated Scikit-Learn pipelines, and XGBoost hyperparameter optimization to balance precision and recall."


1. Project Overview
Goal: Predict the frequency of car insurance claims to assist in risk-based pricing.
This project benchmarks 10 different machine learning classifiers to identify the most robust model for predicting whether a policyholder will file a claim.

2. Data Cleaning & Feature Engineering
Handling Multicollinearity: Used Variance Inflation Factor (VIF) to identify and remove redundant features (e.g., 'red_vehicle'), ensuring the model remains stable for pricing.

Preprocessing Pipeline: Built a custom Scikit-Learn Pipeline to handle:

Imputation: Using KNNImputer for numerical gaps and SimpleImputer for categories.

Encoding: OrdinalEncoder for ranked data (Education) and OneHotEncoder for nominal data (Occupation).

Scaling: Applied StandardScaler to ensure distance-based models (KNN, SVM) performed fairly.

3. Model Benchmarking
I compared 10 different algorithms using 10-Fold Cross-Validation to ensure the results were consistent and not biased by a single data split.

Insert your Boxplot Image here! > Caption: Results show CatBoost and XGBoost as top performers in terms of both median accuracy and stability (low variance).

4. Hyperparameter Optimization
Algorithm: XGBoost (chosen for its balance of speed and high predictive power).

Method: RandomizedSearchCV with 2,000 iterations.

Metric: Weighted F1-Score. (Explain why: In insurance, we must balance catching high-risk drivers without overpricing safe ones).


5. Key Findings for Jitin Jain
1. The "Frequency" Predictors (Red)
5_year_num_of_claims (0.22) & license_points (0.22): These are your strongest predictors. In actuarial terms, "past behavior predicts future behavior." If someone has points on their license or recent claims, their risk profile increases significantly.

single_parent (0.15): This is a classic (and sometimes sensitive) rating factor. From a data perspective, it often correlates with higher stress or more chaotic driving environments (e.g., driving children).

2. The "Stability" Predictors (Blue)
value_of_home (-0.19) & income (-0.15): These are your strongest "Protective" factors. Higher wealth and homeownership often correlate with financial stability and more cautious behavior, leading to fewer claims.

married (-0.13): Statistically, married drivers tend to be "safer" risks, which is why they often get lower premiums.

3. The "Noise" (Grey)
red_vehicle (-0.007): Look at that! It's almost zero. This confirms your earlier decision to drop this column. The "red cars are faster/riskier" idea is a myth in this dataset.
