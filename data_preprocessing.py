"""
Data Preprocessing Module
Handles dataset loading, cleaning, standardization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict
import json


class DataPreprocessor:
    """Handles all data preprocessing tasks for the diagnosis prediction system"""
    
    def __init__(self):
        self.symptom_encoder = MultiLabelBinarizer()
        self.diagnosis_encoder = LabelEncoder()
        self.symptom_vocabulary = None
        self.severity_mapping = {'mild': 1, 'moderate': 2, 'severe': 3}
        self.duration_mapping = {'<1day': 1, '1-3days': 2, '3-7days': 3, '>7days': 4}
        
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        Expected columns: symptoms, diagnosis, severity (optional), duration (optional)
        """
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Dataset loaded: {len(df)} records")
            return df
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            raise
    
    def create_sample_dataset(self, output_path: str = "medical_data.csv"):
        """
        Creates a realistic sample dataset for demonstration
        In production, you'd use a real medical dataset like:
        - Disease Symptom Prediction Dataset (Kaggle)
        - MIMIC-III (with proper access)
        - Synthetic medical records
        """
        data = {
            'symptoms': [
                'fever,cough,fatigue',
                'fever,sore_throat,runny_nose',
                'chest_pain,shortness_of_breath,sweating',
                'headache,nausea,sensitivity_to_light',
                'abdominal_pain,diarrhea,vomiting',
                'fever,cough,chest_pain',
                'runny_nose,sneezing,sore_throat',
                'chest_pain,dizziness,irregular_heartbeat',
                'severe_headache,stiff_neck,confusion',
                'abdominal_pain,fever,loss_of_appetite',
                'cough,wheezing,shortness_of_breath',
                'fever,chills,body_aches,fatigue',
                'nausea,vomiting,diarrhea,fever',
                'headache,fatigue,dizziness',
                'chest_pain,cough,fever,night_sweats',
                'sore_throat,fever,swollen_glands',
                'shortness_of_breath,chest_tightness,wheezing',
                'headache,visual_disturbances,nausea',
                'fever,rash,joint_pain',
                'abdominal_pain,bloating,constipation',
            ] * 50,  # Repeat to get 1000 samples
            'diagnosis': [
                'Common Cold',
                'Flu',
                'Heart Attack',
                'Migraine',
                'Gastroenteritis',
                'Pneumonia',
                'Allergic Rhinitis',
                'Cardiac Arrhythmia',
                'Meningitis',
                'Appendicitis',
                'Asthma',
                'Influenza',
                'Food Poisoning',
                'Tension Headache',
                'Tuberculosis',
                'Strep Throat',
                'Asthma Attack',
                'Cluster Headache',
                'Dengue Fever',
                'Irritable Bowel Syndrome',
            ] * 50,
            'severity': np.random.choice(['mild', 'moderate', 'severe'], 1000),
            'duration': np.random.choice(['<1day', '1-3days', '3-7days', '>7days'], 1000)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"✓ Sample dataset created: {output_path}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataset"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df['symptoms'] = df['symptoms'].fillna('')
        df['diagnosis'] = df['diagnosis'].fillna('Unknown')
        
        # Standardize text (lowercase, strip whitespace)
        df['symptoms'] = df['symptoms'].str.lower().str.strip()
        df['diagnosis'] = df['diagnosis'].str.strip()
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            print(f"✓ Removed {initial_len - len(df)} duplicate records")
        
        # Remove records with empty symptoms or diagnosis
        df = df[df['symptoms'].str.len() > 0]
        df = df[df['diagnosis'] != 'Unknown']
        
        print(f"✓ Data cleaned: {len(df)} valid records")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert symptoms to feature vectors
        Returns: (X, y) where X is feature matrix and y is encoded diagnoses
        """
        # Parse symptoms (comma-separated) into lists
        symptom_lists = df['symptoms'].str.split(',').apply(lambda x: [s.strip() for s in x])
        
        # Create multi-hot encoded symptom vectors
        X_symptoms = self.symptom_encoder.fit_transform(symptom_lists)
        self.symptom_vocabulary = self.symptom_encoder.classes_
        
        # Add severity and duration features if available
        additional_features = []
        
        if 'severity' in df.columns:
            severity_encoded = df['severity'].map(self.severity_mapping).fillna(0).values.reshape(-1, 1)
            additional_features.append(severity_encoded)
        
        if 'duration' in df.columns:
            duration_encoded = df['duration'].map(self.duration_mapping).fillna(0).values.reshape(-1, 1)
            additional_features.append(duration_encoded)
        
        # Combine all features
        if additional_features:
            X = np.hstack([X_symptoms] + additional_features)
        else:
            X = X_symptoms
        
        # Encode diagnoses
        y = self.diagnosis_encoder.fit_transform(df['diagnosis'])
        
        print(f"✓ Features engineered: {X.shape[1]} features, {len(self.diagnosis_encoder.classes_)} classes")
        print(f"  - Symptoms vocabulary: {len(self.symptom_vocabulary)} unique symptoms")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Dict:
        """
        Split data into train, validation, and test sets
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"✓ Data split:")
        print(f"  - Training: {len(X_train)} samples")
        print(f"  - Validation: {len(X_val)} samples")
        print(f"  - Test: {len(X_test)} samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def balance_classes(self, X: np.ndarray, y: np.ndarray, 
                       method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using SMOTE or other techniques
        """
        if method == 'smote':
            # Check if we have enough samples for SMOTE
            unique, counts = np.unique(y, return_counts=True)
            min_samples = min(counts)
            
            if min_samples < 2:
                print("⚠ Warning: Some classes have too few samples for SMOTE. Skipping balancing.")
                return X, y
            
            # Use k_neighbors based on smallest class
            k_neighbors = min(5, min_samples - 1)
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            print(f"✓ Classes balanced using SMOTE:")
            print(f"  - Before: {len(X)} samples")
            print(f"  - After: {len(X_balanced)} samples")
            
            return X_balanced, y_balanced
        else:
            return X, y
    
    def encode_user_symptoms(self, symptoms: List[str], 
                            severity: str = None, 
                            duration: str = None) -> np.ndarray:
        """
        Encode user-provided symptoms for inference
        """
        # Clean and standardize symptoms
        symptoms = [s.lower().strip() for s in symptoms]
        
        # Create multi-hot vector
        symptom_vector = self.symptom_encoder.transform([symptoms])[0]
        
        # Add severity and duration if provided
        additional_features = []
        
        if severity:
            severity_val = self.severity_mapping.get(severity.lower(), 0)
            additional_features.append(severity_val)
        
        if duration:
            duration_val = self.duration_mapping.get(duration.lower(), 0)
            additional_features.append(duration_val)
        
        # Combine features
        if additional_features:
            feature_vector = np.concatenate([symptom_vector, additional_features])
        else:
            feature_vector = symptom_vector
        
        return feature_vector.reshape(1, -1)
    
    def save_encoders(self, filepath: str = "encoders.json"):
        """Save encoders for later use"""
        encoders_data = {
            'symptom_vocabulary': self.symptom_vocabulary.tolist(),
            'diagnosis_classes': self.diagnosis_encoder.classes_.tolist(),
            'severity_mapping': self.severity_mapping,
            'duration_mapping': self.duration_mapping
        }
        
        with open(filepath, 'w') as f:
            json.dump(encoders_data, f, indent=2)
        
        print(f"✓ Encoders saved to {filepath}")
    
    def load_encoders(self, filepath: str = "encoders.json"):
        """Load previously saved encoders"""
        with open(filepath, 'r') as f:
            encoders_data = json.load(f)
        
        self.symptom_vocabulary = np.array(encoders_data['symptom_vocabulary'])
        self.symptom_encoder = MultiLabelBinarizer()
        self.symptom_encoder.fit([self.symptom_vocabulary.tolist()])
        
        self.diagnosis_encoder = LabelEncoder()
        self.diagnosis_encoder.classes_ = np.array(encoders_data['diagnosis_classes'])
        
        self.severity_mapping = encoders_data['severity_mapping']
        self.duration_mapping = encoders_data['duration_mapping']
        
        print(f"✓ Encoders loaded from {filepath}")


if __name__ == "__main__":
    # Demo usage
    print("=== Data Preprocessing Demo ===\n")
    
    preprocessor = DataPreprocessor()
    
    # Create sample dataset
    df = preprocessor.create_sample_dataset()
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Engineer features
    X, y = preprocessor.engineer_features(df)
    
    # Split data
    data_split = preprocessor.split_data(X, y)
    
    # Balance training data
    X_train_balanced, y_train_balanced = preprocessor.balance_classes(
        data_split['X_train'], 
        data_split['y_train']
    )
    
    # Save encoders
    preprocessor.save_encoders()
    
    print("\n✓ Preprocessing complete!")
