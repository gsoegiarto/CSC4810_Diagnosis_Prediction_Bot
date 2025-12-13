import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import json
from datetime import datetime

print("="*70)
print("TRAINING DIAGNOSIS MODEL")
print("="*70)

# Step 1: Load data
print("\n1. Loading data...")
df = pd.read_csv('medical_data.csv')
print(f"   ✓ Loaded {len(df)} records")
print(f"   ✓ Diseases: {df['diagnosis'].nunique()}")

# Step 2: Process symptoms
print("\n2. Processing symptoms...")

# Get all unique symptoms
all_symptoms = set()
for symptoms_str in df['symptoms']:
    symptoms = str(symptoms_str).split(',')
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom:
            all_symptoms.add(symptom)

symptom_list = sorted(list(all_symptoms))
print(f"   ✓ Found {len(symptom_list)} unique symptoms")

# Create feature matrix
def create_features(symptoms_str, symptom_list):
    """Convert symptom string to binary feature vector"""
    symptoms = [s.strip() for s in str(symptoms_str).split(',')]
    features = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    return features

# Build feature matrix
X = []
y = []

for idx, row in df.iterrows():
    features = create_features(row['symptoms'], symptom_list)
    X.append(features)
    y.append(row['diagnosis'])

X = np.array(X)
y = np.array(y)

print(f"   ✓ Feature matrix: {X.shape}")

# Step 3: Encode labels
print("\n3. Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

classes = label_encoder.classes_.tolist()
print(f"   ✓ {len(classes)} disease classes")

# Step 4: Split data
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   ✓ Training: {len(X_train)} samples")
print(f"   ✓ Testing: {len(X_test)} samples")

# Step 5: Train model
print("\n5. Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
print("   ✓ Model trained")

# Step 6: Evaluate
print("\n6. Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"   ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   ✓ Precision: {precision:.4f}")
print(f"   ✓ Recall:    {recall:.4f}")
print(f"   ✓ F1 Score:  {f1:.4f}")

# Step 7: Save model
print("\n7. Saving model...")

# Save model
model_data = {
    'model': model,
    'label_encoder': label_encoder,
    'symptom_vocabulary': symptom_list
}
joblib.dump(model_data, 'diagnosis_model.pkl')
print("   ✓ Model saved to: diagnosis_model.pkl")

# Save encoders
encoders = {
    'symptom_vocabulary': symptom_list,
    'diagnosis_classes': classes
}
with open('encoders.json', 'w') as f:
    json.dump(encoders, f, indent=2)
print("   ✓ Encoders saved to: encoders.json")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(symptom_list),
        'num_classes': len(classes),
        'classes': classes
    },
    'test_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
}

with open('training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("   ✓ Summary saved to: training_summary.json")

# Done!
print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel Performance:")
print(f"  - Test Accuracy: {accuracy*100:.2f}%")
print(f"  - F1 Score: {f1*100:.2f}%")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Disease classes: {len(classes)}")

print(f"\nFiles created:")
print(f"  ✓ diagnosis_model.pkl")
print(f"  ✓ encoders.json")
print(f"  ✓ training_summary.json")

print(f"\nNext step:")
print(f"  Run: python3 web_app.py")
print(f"  Then visit: http://localhost:5001")

print("="*70)