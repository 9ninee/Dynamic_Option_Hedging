# %%
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
# Add Option_Data to sys.path to import the loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Option_Data')))
from load_merged_data import load_merged_option_ohlcv

# %%
# Load merged daily data (one row per date)
# Dynamically find the project root (where .git or Option_Data exists)
def find_project_root(current_path):
    while True:
        if os.path.isdir(os.path.join(current_path, 'Option_Data')) or \
           os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise RuntimeError('Project root not found!')
        current_path = parent

project_root = find_project_root(os.path.abspath(__file__))
merged_folder = os.path.join(project_root, "Option_Data", "Test_data", "merged")
df = load_merged_option_ohlcv(merged_folder=merged_folder)

df.head()
# %%
# Drop rows with missing return (first day or missing data)
df = df.dropna(subset=['return'])

# Set target: 1 if return > 0 else 0
y = (df['return'] > 0).astype(int)

# Use all numeric columns except 'return' as features
feature_cols = df.select_dtypes(include='number').columns.difference(['return','log_return'])
X = df[feature_cols]

# %%
# Chronological train/test split for time series
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Sample features: {X_train.head()}")
print(f"Sample targets: {y_train.head()}")

# %%
# Train and compare multiple models with feature scaling, cross-validation, and overfitting prevention
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, max_depth=5),  # limit depth to prevent overfitting
    'SVM': make_pipeline(StandardScaler(), SVC(random_state=42)),  # scale features
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=3),  # limit depth
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier())  # scale features
}

results = []
for name, model in models.items():
    # Cross-validation (on training set)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results.append((name, cv_scores.mean(), test_acc))

# Print results in a table
print("\nModel Comparison (with CV and Test Accuracy):")
print("{:<20} {:<15} {:<10}".format('Model', 'CV Accuracy', 'Test Acc'))
for name, cv_acc, test_acc in results:
    print("{:<20} {:.4f}         {:.4f}".format(name, cv_acc, test_acc))

# %%
# Show feature importances for models that support it
import numpy as np

print("\nFeature Importances:")
feature_names = list(feature_cols)

# Logistic Regression (absolute value of coefficients)
lr_model = models['Logistic Regression']
if hasattr(lr_model, 'coef_'):
    importances = np.abs(lr_model.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]
    print("\nLogistic Regression (|coefficients|):")
    print("{:<25} {:<10}".format('Feature', 'Importance'))
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]:<25} {importances[idx]:.4f}")

# Random Forest
rf_model = models['Random Forest']
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nRandom Forest:")
    print("{:<25} {:<10}".format('Feature', 'Importance'))
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]:<25} {importances[idx]:.4f}")

# Gradient Boosting
gb_model = models['Gradient Boosting']
if hasattr(gb_model, 'feature_importances_'):
    importances = gb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nGradient Boosting:")
    print("{:<25} {:<10}".format('Feature', 'Importance'))
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]:<25} {importances[idx]:.4f}")

# SVM and KNN do not provide feature importances
print("\nSVM and KNN do not provide feature importances.")
# %%
