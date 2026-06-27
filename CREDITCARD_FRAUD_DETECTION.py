import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings('ignore')

DATA_FILE = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f'Dataset not found: {DATA_FILE}')

# Load dataset
print('Loading dataset from:', DATA_FILE)
df = pd.read_csv(DATA_FILE)
print('Dataset shape:', df.shape)
print(df['Class'].value_counts())

# Separate features and target
y = df['Class']
X = df.drop(columns=['Class'])

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split with stratification to preserve class imbalance in test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print('\nTraining class distribution:')
print(pd.Series(y_train).value_counts())
print('Testing class distribution:')
print(pd.Series(y_test).value_counts())

# Build a balanced training set using random oversampling of the minority class
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data['Class'] = y_train.values
majority = train_data[train_data['Class'] == 0]
minority = train_data[train_data['Class'] == 1]
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42,
)
train_balanced = pd.concat([majority, minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = train_balanced.drop(columns=['Class']).values
y_train_balanced = train_balanced['Class'].values

print('\nBalanced training class distribution:')
print(pd.Series(y_train_balanced).value_counts())

# Train logistic regression
log_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
log_model.fit(X_train_balanced, y_train_balanced)
log_pred = log_model.predict(X_test)

# Train random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)

# Evaluation helper
def evaluate_model(name, y_true, y_pred):
    print(f'=== {name} Evaluation ===')
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:   ', recall_score(y_true, y_pred))
    print('F1-score: ', f1_score(y_true, y_pred))
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, digits=4))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('\n')

evaluate_model('Logistic Regression', y_test, log_pred)
evaluate_model('Random Forest', y_test, rf_pred)

results_df = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': rf_pred})

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.show()

# Compare with training on original imbalanced data using class weighting
print('Training Random Forest on original imbalanced data with class_weight="balanced"...')
rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_weighted.fit(X_train, y_train)
rf_weighted_pred = rf_weighted.predict(X_test)
evaluate_model('Random Forest (balanced class_weight)', y_test, rf_weighted_pred)
