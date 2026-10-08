"""
Project 2: Iris Classification
Goal: Compare 3 different classification models on the same dataset
Learn: When to use which model and how to compare them
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("=" * 70)
print("PROJECT 2: IRIS CLASSIFICATION - COMPARING 3 MODELS")
print("=" * 70)

# STEP 1: Load data
print("\n[STEP 1] Loading Iris Dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Samples per class: {np.bincount(y)}")

# STEP 2: Explore data
print("\n[STEP 2] Exploring Data...")
print("\nFirst 5 rows:")
print(X.head())
print("\nData statistics:")
print(X.describe())

# STEP 3: Split data
print("\n[STEP 3] Splitting data into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# STEP 4: Train 3 different models
print("\n[STEP 4] Training 3 Different Models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")

# STEP 5: Compare results
print("\n[STEP 5] Model Comparison...")
print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

for model_name, data in results.items():
    accuracy = data['accuracy']
    bar = "█" * int(accuracy * 50)
    print(f"{model_name:25} | {accuracy:.4f} | {bar}")

best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# STEP 6: Detailed evaluation of best model
print("\n[STEP 6] Detailed Evaluation of Best Model...")
best_model = results[best_model_name]['model']
best_pred = results[best_model_name]['predictions']

print(f"\nConfusion Matrix ({best_model_name}):")
cm = confusion_matrix(y_test, best_pred)
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, 
                          target_names=iris.target_names))

# STEP 7: Visualize comparison
print("\n[STEP 7] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: Model comparison
ax1 = axes[0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
colors = ['green' if acc == max(accuracies) else 'blue' for acc in accuracies]

ax1.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0.9, 1.0])
ax1.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')

for i, (name, acc) in enumerate(zip(model_names, accuracies)):
    ax1.text(i, acc + 0.002, f'{acc:.4f}', ha='center', fontsize=10)

ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Confusion matrix heatmap
ax2 = axes[1]
im = ax2.imshow(cm, cmap='Blues', aspect='auto')
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_title(f'Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(iris.target_names, rotation=45)
ax2.set_yticklabels(iris.target_names)

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, cm[i, j],
                       ha="center", va="center", color="black", fontsize=12)

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
print("✓ Visualization saved as 'model_comparison.png'")

# STEP 8: Key findings
print("\n[STEP 8] Key Findings:")
print(f"1. Best performing model: {best_model_name}")
print(f"2. Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"3. Why {best_model_name}?")
if best_model_name == 'Random Forest':
    print("   - Ensemble method combines multiple trees")
    print("   - Reduces overfitting through averaging")
    print("   - Works well with small datasets")
elif best_model_name == 'Decision Tree':
    print("   - Simple, interpretable model")
    print("   - Captures non-linear relationships")
    print("   - No feature scaling needed")
else:
    print("   - Works well for linearly separable data")
    print("   - Fast training and prediction")
    print("   - Probability estimates included")

print(f"4. All models perform well (>90% accuracy on Iris)")
print(f"5. Iris is an easy dataset - try harder datasets next")

print("\n" + "=" * 70)
print("PROJECT 2 COMPLETE!")
print("=" * 70)