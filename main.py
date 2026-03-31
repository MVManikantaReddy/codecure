import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_excel('Dataset.xlsx')

print(f"Dataset shape: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

print("\nMissing values per column:")
print(df.isnull().sum())

target_col = None
for col in df.columns:
    vals = df[col].astype(str).str.lower()
    if vals.str.contains('resistant|sensitive|intermediate', na=False).any():
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f"\nTarget column not detected automatically.")
    print(f"Using last column '{target_col}' as target.")
else:
    print(f"\nTarget column found: '{target_col}'")

df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col].notna()]
df = df[df[target_col].astype(str).str.strip() != ""]

le = LabelEncoder()
y = le.fit_transform(df[target_col])

print(f"\nTarget classes: {list(le.classes_)}")
print("Class counts:")
print(df[target_col].value_counts())

feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols].copy()

X = X.dropna(axis=1, how='all')
feature_cols = list(X.columns)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]

if len(num_cols) > 0:
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

if len(cat_cols) > 0:
    X[cat_cols] = X[cat_cols].astype(str).fillna("Unknown")

encoder = None
if len(cat_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=le.classes_))

print("\nFeature Importance")
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df.to_string(index=False))
importance_df.to_csv('feature_importance.csv', index=False)

plt.figure(figsize=(10, 6))
top_features = importance_df.head(10)
sns.barplot(data=top_features, x='Importance', y='Feature', palette='coolwarm')
plt.title('Top 10 Features Predicting Antibiotic Resistance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: feature_importance.png")

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: confusion_matrix.png")

def predict(sample):
    sample_df = pd.DataFrame([sample])

    for col in X.columns:
        if col not in sample_df.columns:
            sample_df[col] = np.nan

    sample_df = sample_df[X.columns]

    if len(num_cols) > 0:
        for col in num_cols:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')
        sample_df[num_cols] = sample_df[num_cols].fillna(X[num_cols].median())

    if len(cat_cols) > 0:
        sample_df[cat_cols] = sample_df[cat_cols].astype(str).fillna("Unknown")
        sample_df[cat_cols] = encoder.transform(sample_df[cat_cols])

    pred = model.predict(sample_df)[0]
    return le.inverse_transform([pred])[0]

print("\nModel is ready! Use predict({...}) for new samples")

print("\nAll files generated successfully!")
print("Files created:")
print("1. main.py")
print("2. feature_importance.csv")
print("3. feature_importance.png")
print("4. confusion_matrix.png")