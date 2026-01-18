import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#sklearn preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

#Classification Model

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

#1. Data Loading
df = pd.read_csv("E:\\ML Phitron\\ML Assignment\\test_folder\\loan_approval_dataset.csv")
print(df.shape)
print(df.head(10))

#2. Data Preprocessing
df.columns = df.columns.str.strip()
df.drop(columns=["no_of_dependents", "loan_id"], inplace=True)
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].replace(0, np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

df.loc[df["residential_assets_value"] < 0, "residential_assets_value"] = np.nan
df["residential_assets_value"] = df["residential_assets_value"].fillna(df["residential_assets_value"].median())


df["loan_status_enc"] = ( df["loan_status"].astype(str).str.strip().map({"Rejected": 0, "Approved": 1}))
X = df.drop(columns=["loan_status", "loan_status_enc"])
y = df["loan_status_enc"]

#3. Pipeline Creation
numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns
num_transformer = Pipeline (
    steps = [
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
cat_transformer = Pipeline( steps = [
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=False))
] )
preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numeric_features),
        ('cat',cat_transformer,categorical_features)
    ]
    )

#4. Primary Model Selection
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", xgb)
])


#5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42, stratify=y)

xgb_pipe.fit(X_train, y_train)

y_pred = xgb_pipe.predict(X_test)
y_prob = xgb_pipe.predict_proba(X_test)[:, 1]

#6. Cross-Validation
cv_scores = cross_val_score( xgb_pipe, X_train, y_train, cv=5, n_jobs = -1, scoring="f1" )
print("Average CV score: and Std (robustness):\n")
print("Average CV score:", cv_scores.mean())
print("Std (robustness):", cv_scores.std())

#7. Hyperparameter Tuning
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
}
grid = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=2,
    random_state=42
)


#8. Best Model Selection
grid.fit(X_train, y_train)
random_search.fit(X_train,y_train)

print("Grid Search Result: \n")

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("\n")
print("Randomized Search Result: \n")
print( random_search.best_params_ )
print( random_search.best_score_ )


#9. Model Performance Evaluation
best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test)

randomized_model = random_search.best_estimator_
y_pred_randomized = randomized_model.predict(X_test)

print("Normal Accuracy and CL report:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("CL Report:\n",classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC (Average Precision):", average_precision_score(y_test, y_prob))
print("\nBest Model Selection Accuracy and CL report:\n")
print("Test Accuracy (Tuned):", accuracy_score(y_test, y_pred_tuned))
print("CL Report:\n", classification_report(y_test, y_pred_tuned))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))


with open("loan_approval_XGBoost_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("XGBoost Pipeline saved as loan_approval_XGBoost_model.pkl")

