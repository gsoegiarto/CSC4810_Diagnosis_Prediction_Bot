import pandas as pd
import numpy as np

print("="*70)
print("PROCESSING KAGGLE DISEASE-SYMPTOM DATASET")
print("="*70)

# Load the Kaggle dataset
print("\nLoading dataset.csv...")
df = pd.read_csv('dataset.csv')
print(f"✓ Loaded {len(df)} records")

# Process the data
processed_data = []

for idx, row in df.iterrows():
    # Get disease name
    disease = row['Disease']
    
    # Collect symptoms from columns Symptom_1 to Symptom_17
    symptoms = []
    for i in range(1, 18):
        col_name = f'Symptom_{i}'
        if col_name in df.columns and pd.notna(row[col_name]):
            symptom = str(row[col_name]).strip().lower().replace(' ', '_')
            if symptom and symptom != 'nan':
                symptoms.append(symptom)
    
    # Only add if we have symptoms
    if len(symptoms) > 0:
        # Assign severity and duration based on number of symptoms
        num_symptoms = len(symptoms)
        if num_symptoms >= 10:
            severity = 'severe'
            duration = '>7days'
        elif num_symptoms >= 6:
            severity = 'moderate'
            duration = '3-7days'
        else:
            severity = 'mild'
            duration = '1-3days'
        
        processed_data.append({
            'symptoms': ','.join(symptoms),
            'diagnosis': disease,
            'severity': severity,
            'duration': duration
        })

# Create DataFrame and save
result_df = pd.DataFrame(processed_data)
result_df.to_csv('medical_data.csv', index=False)

print(f"\n✓ Processed {len(result_df)} records")
print(f"✓ Unique diseases: {result_df['diagnosis'].nunique()}")
print(f"✓ Unique symptom combinations: {result_df['symptoms'].nunique()}")

print("\n" + "="*70)
print("✓ SAVED TO: medical_data.csv")
print("="*70)

print("\n📊 Disease Distribution (top 10):")
disease_counts = result_df['diagnosis'].value_counts().head(10)
for disease, count in disease_counts.items():
    print(f"   - {disease}: {count} cases")

print("\n" + "="*70)
print("NEXT STEP: Run 'python3 retrain_model.py'")
print("="*70)