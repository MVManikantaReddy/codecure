# CodeCure AI Hackathon - Track B
# Antibiotic Resistance Prediction using Random Forest ML
# Author: MVManikantaReddy, B.Tech AI/ML, Aditya University

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CODECURE AI HACKATHON - TRACK B")
print("Antibiotic Resistance Prediction (Random Forest ML)")
print("="*60)
print()

# Generate synthetic dataset (1000 patients)
np.random.seed(42)
n = 1000

print("Generating synthetic dataset...")
Age = np.random.randint(18, 85, n)
Gender = np.random.choice(['Male', 'Female'], n)
Specimen_Type = np.random.choice(['Urine', 'Blood', 'Sputum', 'Wound'], n, p=[0.4, 0.25, 0.25, 0.1])
Antibiotic_Type = np.random.choice(['Ciprofloxacin', 'Amoxicillin', 'Levofloxacin', 'Gentamicin'], n)
CRP_level = np.random.uniform(1, 150, n)
WBC_count = np.random.uniform(4000, 20000, n)
Creatinine = np.random.uniform(0.6, 5.0, n)

# Antibiotic Resistance based on features (Age most important ~62%)
base_risk = 0.3 + (Age - 40) / 100 + (CRP_level - 50) / 300 + (WBC_count - 8000) / 20000
risk_prob = np.clip(base_risk + np.random.normal(0, 0.1, n), 0, 1)
Antibiotic_Resistance = (risk_prob > 0.45).astype(int)

df = pd.DataFrame({
    'Age': Age,
    'Gender': Gender,
    'Specimen_Type': Specimen_Type,
    'Antibiotic_Type': Antibiotic_Type,
    'CRP_level': CRP_level,
    'WBC_count': WBC_count,
    'Creatinine': Creatinine,
    'Antibiotic_Resistance': Antibiotic_Resistance
})

print(f"Dataset generated: {len(df)} patients")
print()
print("Dataset Info:")
print(df.head())
print()
print(f"Dataset shape: {df.shape}")
print()


# Preprocessing: Encode categorical variables
print("Preprocessing categorical variables...")
df_encoded = pd.get_dummies(df, columns=['Gender', 'Specimen_Type', 'Antibiotic_Type'], drop_first=True)
print(f"Features after encoding: {df_encoded.shape[1]-1}")
print()

# Define features and target
X = df_encoded.drop('Antibiotic_Resistance', axis=1)
y = df_encoded['Antibiotic_Resistance']

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {len(y)}")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print()

# Train Random Forest Model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
print("Model training complete!")
print()

# Training Accuracy
train_pred = rf_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_acc*100:.2f}%")

# Test Accuracy
test_pred = rf_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print()


# Feature Importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print()
print("Top 10 Most Important Features:")
for i, row in feature_imp.head(10).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Importance']*100:.1f}%")
print()

# Visualization: Feature Importance Plot
print("Generating feature_importance.png...")
plt.figure(figsize=(12, 8))
top_features = feature_imp.head(10)
plt.barh(range(len(top_features)), top_features['Importance']*100, align='center')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance (%)')
plt.title('Top 10 Feature Importance - Antibiotic Resistance Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved: feature_importance.png")
print()

# Visualization: Confusion Matrix
print("Generating confusion_matrix.png...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Resistant', 'Susceptible'],
            yticklabels=['Resistant', 'Susceptible'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Antibiotic Resistance Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("Saved: confusion_matrix.png")
print()

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, test_pred, target_names=['Resistant', 'Susceptible']))
print()

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Dataset: {len(df)} patients")
print(f"Model: Random Forest (100 trees, max_depth=10)")
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Top Feature: {feature_imp.iloc[0]['Feature']} ({feature_imp.iloc[0]['Importance']*100:.1f}%)")
print("="*60)
print("Generated files: feature_importance.png, confusion_matrix.png")
print("="*60)
print("\nProject by: MVManikantaReddy")
print("B.Tech AI/ML, Aditya University")
print("CodeCure AI Hackathon - Track B")
