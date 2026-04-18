import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay

# 1. Load and explore dataset
print("Loading dataset...")
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Dataset Shape: {df.shape}")
print("Basic Info:")
print(df.info())
print("\nMissing Values Before:")
print(df.isnull().sum())

# 2. Data preprocessing
# TotalCharges is object, it has some blank spaces ' ' instead of nulls.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Handle missing values (simple method: median)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 3. Feature selection & Engineering
# Select a simple subset of features
selected_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                     'tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']

X = df[selected_features].copy()
y = df['Churn'].copy()

# Convert categorical columns
categorical_cols = ['gender', 'Partner', 'Dependents', 'InternetService', 'Contract']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Target encoding
y_le = LabelEncoder()
y = y_le.fit_transform(y)

# 4. Split dataset (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 5. Train Model
# Logistic Regression (Mandatory)
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)

# Decision Tree (Optional)
print("Training Decision Tree...")
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# 6. Evaluation
y_pred_log = log_reg.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_log):.4f}")

print("\n--- Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_tree):.4f}")

# 7. Visualization
# Plot confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig('confusion_matrix.png')
print("\nSaved confusion_matrix.png")

# Plot Feature Importance from Decision Tree
importance = tree_clf.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(selected_features, importance, color='skyblue')
plt.title("Decision Tree - Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature_importance.png")

# Save the best model (Logistic Regression) and encoders
with open('log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

print("\nModel saved as log_reg_model.pkl")
