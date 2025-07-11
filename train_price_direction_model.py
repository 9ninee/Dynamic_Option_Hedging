# %%
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor

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
# Train and compare multiple models with proper time series cross-validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Use TimeSeriesSplit for proper time series CV (prevents data leakage)
tscv = TimeSeriesSplit(n_splits=5)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, max_depth=5),  # limit depth to prevent overfitting
    'SVM': make_pipeline(StandardScaler(), SVC(random_state=42)),  # scale features
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=3),  # limit depth
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),  # scale features
    'XGBoost Classifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = []
for name, model in models.items():
    # Time series cross-validation (respects temporal order - no data leakage)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test on held-out test set
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    results.append((name, cv_scores.mean(), cv_scores.std(), test_acc))

# Print results in a table
print("\nModel Comparison (Time Series CV - No Data Leakage):")
print("{:<20} {:<15} {:<15} {:<10}".format('Model', 'CV Accuracy', 'CV Std', 'Test Acc'))
for name, cv_acc, cv_std, test_acc in results:
    print("{:<20} {:.4f}±{:.4f}    {:.4f}".format(name, cv_acc, cv_std, test_acc))

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

# SVM and KNN do not provide feature importances.
print("\nSVM and KNN do not provide feature importances.")

# %%
# Additional time series analysis
print("\n=== TIME SERIES ANALYSIS ===")
print("Time Series Cross-Validation Details:")
print(f"Number of splits: {tscv.n_splits}")
print("Each split respects temporal order - no future data used to predict past")

# Show the splits for verification
for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"Split {i+1}: Train size {len(train_idx)}, Validation size {len(val_idx)}")
    print(f"  Train dates: {X_train.index[train_idx[0]]} to {X_train.index[train_idx[-1]]}")
    print(f"  Val dates: {X_train.index[val_idx[0]]} to {X_train.index[val_idx[-1]]}")

# %%
# Add a new target: realized volatility over the next 5 days (rolling std of log returns)
vol_window = 5
# Shift -vol_window so that for each day, the target is the volatility of the NEXT 5 days
future_log_return = df['log_return'].shift(-1)
df['future_volatility'] = df['log_return'].rolling(window=vol_window).std().shift(-vol_window+1)

# Align features and volatility target
# Remove 'log_return' from features for volatility prediction
vol_feature_cols = feature_cols.difference(['log_return'])
vol_X = df[vol_feature_cols]
vol_y = df['future_volatility']
# Drop last vol_window-1 rows (where future volatility is NaN)
vol_X = vol_X.iloc[:-vol_window+1]
vol_y = vol_y.iloc[:-vol_window+1]

# Split for time series (same as before)
vol_split_idx = int(len(vol_X) * 0.8)
vol_X_train, vol_X_test = vol_X.iloc[:vol_split_idx], vol_X.iloc[vol_split_idx:]
vol_y_train, vol_y_test = vol_y.iloc[:vol_split_idx], vol_y.iloc[vol_split_idx:]

# Train a regression model for volatility prediction
vol_reg = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)
xgb_reg = XGBRegressor(random_state=42)
vol_reg.fit(vol_X_train, vol_y_train)
vol_pred = vol_reg.predict(vol_X_test)
xgb_pred = xgb_reg.fit(vol_X_train, vol_y_train).predict(vol_X_test)

# Evaluate regression
mse = mean_squared_error(vol_y_test, vol_pred)
r2 = r2_score(vol_y_test, vol_pred)
xgb_mse = mean_squared_error(vol_y_test, xgb_pred)
xgb_r2 = r2_score(vol_y_test, xgb_pred)
print("\nVolatility Prediction (RandomForestRegressor, next 5-day realized volatility):")
print(f"Test MSE: {mse:.6f}")
print(f"Test R^2: {r2:.4f}")
print("Sample predicted vs actual volatility:")
print(pd.DataFrame({'Predicted': vol_pred[:10], 'Actual': vol_y_test.values[:10]}))
print("\nVolatility Prediction (XGBoostRegressor, next 5-day realized volatility):")
print(f"Test MSE: {xgb_mse:.6f}")
print(f"Test R^2: {xgb_r2:.4f}")
print("Sample predicted vs actual volatility:")
print(pd.DataFrame({'Predicted': xgb_pred[:10], 'Actual': vol_y_test.values[:10]}))

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# --- LSTM for Direction Classification ---
print("\n=== LSTM Direction Classification ===")
seq_len = 10  # Use past 10 days as input sequence
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X)

# Build sequences for classification
dir_X_seq = []
dir_y_seq = []
for i in range(seq_len, len(X_cls_scaled)):
    dir_X_seq.append(X_cls_scaled[i-seq_len:i])
    dir_y_seq.append(y.iloc[i])
dir_X_seq, dir_y_seq = np.array(dir_X_seq), np.array(dir_y_seq)

# Train/test split
split_idx_cls = int(len(dir_X_seq) * 0.8)
dir_X_train, dir_X_test = dir_X_seq[:split_idx_cls], dir_X_seq[split_idx_cls:]
dir_y_train, dir_y_test = dir_y_seq[:split_idx_cls], dir_y_seq[split_idx_cls:]

# Build LSTM model for classification
lstm_cls = Sequential([
    LSTM(32, input_shape=(seq_len, dir_X_seq.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_cls.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_cls.fit(dir_X_train, dir_y_train, epochs=10, batch_size=16, validation_data=(dir_X_test, dir_y_test), verbose=2)
# Predict and evaluate
lstm_cls_pred = (lstm_cls.predict(dir_X_test) > 0.5).astype(int).flatten()
from sklearn.metrics import accuracy_score, classification_report
print('LSTM Direction Test Accuracy:', accuracy_score(dir_y_test, lstm_cls_pred))
print(classification_report(dir_y_test, lstm_cls_pred))

# --- LSTM for Volatility Regression ---
print("\n=== LSTM Volatility Regression ===")
seq_len_reg = 10
scaler_reg = StandardScaler()
vol_X_scaled = scaler_reg.fit_transform(vol_X)

# Build sequences for regression
reg_X_seq = []
reg_y_seq = []
for i in range(seq_len_reg, len(vol_X_scaled)):
    reg_X_seq.append(vol_X_scaled[i-seq_len_reg:i])
    reg_y_seq.append(vol_y.iloc[i])
reg_X_seq, reg_y_seq = np.array(reg_X_seq), np.array(reg_y_seq)

# Train/test split
split_idx_reg = int(len(reg_X_seq) * 0.8)
reg_X_train, reg_X_test = reg_X_seq[:split_idx_reg], reg_X_seq[split_idx_reg:]
reg_y_train, reg_y_test = reg_y_seq[:split_idx_reg], reg_y_seq[split_idx_reg:]

# Build LSTM model for regression
lstm_reg = Sequential([
    LSTM(32, input_shape=(seq_len_reg, reg_X_seq.shape[2])),
    Dropout(0.2),
    Dense(1)
])
lstm_reg.compile(optimizer='adam', loss='mse')
lstm_reg.fit(reg_X_train, reg_y_train, epochs=10, batch_size=16, validation_data=(reg_X_test, reg_y_test), verbose=2)
# Predict and evaluate
lstm_reg_pred = lstm_reg.predict(reg_X_test).flatten()
from sklearn.metrics import mean_squared_error, r2_score
print('LSTM Volatility Test MSE:', mean_squared_error(reg_y_test, lstm_reg_pred))
print('LSTM Volatility Test R2:', r2_score(reg_y_test, lstm_reg_pred))

# %%
