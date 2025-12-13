"""
Model Training Module
Trains Random Forest classifier with probability calibration
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import json


class DiagnosisModel:
    """Random Forest classifier for diagnosis prediction with calibration"""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 20, 
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Initialize the diagnosis prediction model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
        """
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all CPU cores
        )
        
        self.calibrated_model = None
        self.is_trained = False
        self.is_calibrated = False
        self.feature_importances = None
        self.class_names = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              class_names: List[str] = None) -> Dict:
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_names: List of class names for reference
            
        Returns:
            Training metrics dictionary
        """
        print("Training Random Forest model...")
        
        self.base_model.fit(X_train, y_train)
        self.is_trained = True
        self.class_names = class_names
        
        # Store feature importances
        self.feature_importances = self.base_model.feature_importances_
        
        # Get training predictions
        y_train_pred = self.base_model.predict(X_train)
        y_train_proba = self.base_model.predict_proba(X_train)
        
        # Calculate training metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'log_loss': log_loss(y_train, y_train_proba)
        }
        
        print(f"✓ Model trained successfully")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray, 
                  method: str = 'isotonic') -> Dict:
        """
        Calibrate probability predictions using validation set
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibration metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calibration")
        
        print(f"Calibrating model using {method} method...")
        
        # Get uncalibrated predictions
        y_val_proba_uncalibrated = self.base_model.predict_proba(X_val)
        uncalibrated_log_loss = log_loss(y_val, y_val_proba_uncalibrated)
        
        # Apply calibration
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, 
            method=method, 
            cv='prefit'  # Use pre-trained model
        )
        self.calibrated_model.fit(X_val, y_val)
        self.is_calibrated = True
        
        # Get calibrated predictions
        y_val_proba_calibrated = self.calibrated_model.predict_proba(X_val)
        calibrated_log_loss = log_loss(y_val, y_val_proba_calibrated)
        
        print(f"✓ Model calibrated successfully")
        print(f"  - Uncalibrated log loss: {uncalibrated_log_loss:.4f}")
        print(f"  - Calibrated log loss: {calibrated_log_loss:.4f}")
        print(f"  - Improvement: {((uncalibrated_log_loss - calibrated_log_loss) / uncalibrated_log_loss * 100):.2f}%")
        
        return {
            'uncalibrated_log_loss': uncalibrated_log_loss,
            'calibrated_log_loss': calibrated_log_loss,
            'improvement_pct': (uncalibrated_log_loss - calibrated_log_loss) / uncalibrated_log_loss * 100
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 class_names: List[str] = None) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
            
        Returns:
            Comprehensive evaluation metrics
        """
        model = self.calibrated_model if self.is_calibrated else self.base_model
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model on test set...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'log_loss': log_loss(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Per-class metrics
        if class_names:
            report = classification_report(y_test, y_pred, target_names=class_names, 
                                          output_dict=True, zero_division=0)
            metrics['per_class_metrics'] = report
        
        print(f"✓ Evaluation complete")
        print(f"  - Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Test F1 Score: {metrics['f1']:.4f}")
        print(f"  - Test Log Loss: {metrics['log_loss']:.4f}")
        
        return metrics
    
    def predict_top_k(self, X: np.ndarray, k: int = 3, 
                      class_names: List[str] = None) -> List[List[Tuple[str, float]]]:
        """
        Predict top K most likely conditions with probabilities
        
        Args:
            X: Input features
            k: Number of top predictions to return
            class_names: List of class names
            
        Returns:
            List of top K predictions for each sample [(condition, probability), ...]
        """
        model = self.calibrated_model if self.is_calibrated else self.base_model
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get probability predictions
        probabilities = model.predict_proba(X)
        
        results = []
        for proba in probabilities:
            # Get top K indices
            top_k_indices = np.argsort(proba)[::-1][:k]
            top_k_probs = proba[top_k_indices]
            
            # Create list of (condition, probability) tuples
            if class_names:
                predictions = [(class_names[idx], prob) for idx, prob in zip(top_k_indices, top_k_probs)]
            else:
                predictions = [(f"Class_{idx}", prob) for idx, prob in zip(top_k_indices, top_k_probs)]
            
            results.append(predictions)
        
        return results
    
    def get_feature_importance(self, feature_names: List[str] = None, 
                               top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important features
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importances is None:
            raise ValueError("Model must be trained to get feature importances")
        
        # Get top N indices
        top_indices = np.argsort(self.feature_importances)[::-1][:top_n]
        
        if feature_names:
            top_features = [(feature_names[idx], self.feature_importances[idx]) 
                           for idx in top_indices]
        else:
            top_features = [(f"Feature_{idx}", self.feature_importances[idx]) 
                           for idx in top_indices]
        
        return top_features
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray, 
                             class_names: List[str], 
                             save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def plot_feature_importance(self, feature_names: List[str], 
                               top_n: int = 20,
                               save_path: str = "feature_importance.png"):
        """Plot and save feature importance"""
        top_features = self.get_feature_importance(feature_names, top_n)
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved to {save_path}")
    
    def save_model(self, filepath: str = "diagnosis_model.pkl"):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'base_model': self.base_model,
            'calibrated_model': self.calibrated_model,
            'is_calibrated': self.is_calibrated,
            'feature_importances': self.feature_importances,
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = "diagnosis_model.pkl"):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.base_model = model_data['base_model']
        self.calibrated_model = model_data['calibrated_model']
        self.is_calibrated = model_data['is_calibrated']
        self.feature_importances = model_data['feature_importances']
        self.class_names = model_data['class_names']
        self.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Train and compare baseline models for reference
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    
    print("\n=== Training Baseline Models for Comparison ===\n")
    
    models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        logloss = log_loss(y_test, y_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'log_loss': logloss
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Log Loss: {logloss:.4f}")
    
    print("\n✓ Baseline comparison complete")
    return results


if __name__ == "__main__":
    # Demo usage
    print("=== Model Training Demo ===\n")
    
    # This would normally use real data from data_preprocessing.py
    # For demo, we'll create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    n_classes = 10
    
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_val = np.random.rand(200, n_features)
    y_val = np.random.randint(0, n_classes, 200)
    X_test = np.random.rand(200, n_features)
    y_test = np.random.randint(0, n_classes, 200)
    
    class_names = [f"Condition_{i}" for i in range(n_classes)]
    feature_names = [f"Symptom_{i}" for i in range(n_features)]
    
    # Train model
    model = DiagnosisModel(n_estimators=100, max_depth=15)
    train_metrics = model.train(X_train, y_train, class_names)
    
    # Calibrate model
    calib_metrics = model.calibrate(X_val, y_val, method='isotonic')
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test, class_names)
    
    # Get top 3 predictions
    predictions = model.predict_top_k(X_test[:5], k=3, class_names=class_names)
    print("\n=== Sample Predictions ===")
    for i, preds in enumerate(predictions):
        print(f"\nSample {i+1}:")
        for j, (condition, prob) in enumerate(preds, 1):
            print(f"  {j}. {condition}: {prob:.4f}")
    
    # Save model
    model.save_model()
    
    print("\n✓ Training pipeline complete!")
