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

# Load dataset
print("Loading dataset...")
df = pd.read_excel("code cure project.xlsx")
print(f"Dataset loaded: {len(df)} patients")
print()

# Display basic info
print("Dataset Info:")
print(df.head())
print()
print(f"Dataset shape: {df.shape}")
print()


# Check for missing values
print("Missing values:")
print(df.isnull().sum())
print()

# Check target variable distribution
print("Target Variable Distribution (Antibiotic Resistance):")
print(df['Antibiotic_Resistance'].value_counts())
print()

# Preprocessing: Encode categorical variables
print("Preprocessing categorical variables...")
df_encoded = pd.get_dummies(df, columns=['Specimen_Type', 'Antibiotic_Type'], drop_first=True)
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
