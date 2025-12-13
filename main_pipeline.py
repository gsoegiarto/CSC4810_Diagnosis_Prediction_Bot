import numpy as np
from data_preprocessing import DataPreprocessor
from model_training import DiagnosisModel, train_baseline_models
from inference_pipeline import InferencePipeline, BatchInference
from safety_module import SafetyChecker
import json
from datetime import datetime


def run_complete_pipeline(use_sample_data: bool = True, 
                         data_path: str = None,
                         balance_data: bool = True):
    
    print("=" * 80)
    print("DIAGNOSIS PREDICTION BOT - COMPLETE ML PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    # STEP 1: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80 + "\n")
    
    preprocessor = DataPreprocessor()
    
    if use_sample_data:
        print("Creating sample dataset...")
        df = preprocessor.create_sample_dataset("medical_data.csv")
    else:
        if not data_path:
            raise ValueError("data_path must be provided when use_sample_data=False")
        print(f"Loading dataset from {data_path}...")
        df = preprocessor.load_dataset(data_path)
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Engineer features
    X, y = preprocessor.engineer_features(df)
    
    # Split data
    data_split = preprocessor.split_data(X, y, test_size=0.2, val_size=0.1)
    
    X_train = data_split['X_train']
    y_train = data_split['y_train']
    X_val = data_split['X_val']
    y_val = data_split['y_val']
    X_test = data_split['X_test']
    y_test = data_split['y_test']
    
    # Balance training data (optional)
    if balance_data:
        X_train, y_train = preprocessor.balance_classes(X_train, y_train, method='smote')
    
    # Save encoders for inference
    preprocessor.save_encoders("encoders.json")
    
    class_names = preprocessor.diagnosis_encoder.classes_.tolist()
    feature_names = list(preprocessor.symptom_vocabulary)
    if 'severity' in df.columns:
        feature_names.append('severity')
    if 'duration' in df.columns:
        feature_names.append('duration')
    
    print(f"\n✓ Data preprocessing complete")
    print(f"  - Total features: {X_train.shape[1]}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Number of classes: {len(class_names)}")
    
    # ========================================================================
    # STEP 2: BASELINE MODEL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: BASELINE MODEL COMPARISON")
    print("=" * 80 + "\n")
    
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # ========================================================================
    # STEP 3: TRAIN MAIN MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN RANDOM FOREST MODEL")
    print("=" * 80 + "\n")
    
    model = DiagnosisModel(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    
    train_metrics = model.train(X_train, y_train, class_names)
    
    # ========================================================================
    # STEP 4: CALIBRATE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: PROBABILITY CALIBRATION")
    print("=" * 80 + "\n")
    
    calib_metrics = model.calibrate(X_val, y_val, method='isotonic')
    
    # ========================================================================
    # STEP 5: EVALUATE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 80 + "\n")
    
    test_metrics = model.evaluate(X_test, y_test, class_names)
    
    # Print detailed metrics
    print("\n--- Detailed Test Metrics ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Log Loss: {test_metrics['log_loss']:.4f}")
    
    # Plot visualizations
    print("\n--- Generating Visualizations ---")
    model.plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        class_names, 
        "confusion_matrix.png"
    )
    
    model.plot_feature_importance(
        feature_names, 
        top_n=20, 
        save_path="feature_importance.png"
    )
    
    # ========================================================================
    # STEP 6: SAVE MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVE MODEL")
    print("=" * 80 + "\n")
    
    model.save_model("diagnosis_model.pkl")
    
    # Save training summary
    training_summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1],
            'num_classes': len(class_names),
            'classes': class_names
        },
        'model_config': {
            'type': 'RandomForest',
            'n_estimators': 200,
            'max_depth': 20,
            'calibrated': True,
            'calibration_method': 'isotonic'
        },
        'baseline_results': baseline_results,
        'training_metrics': train_metrics,
        'calibration_metrics': calib_metrics,
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'log_loss': float(test_metrics['log_loss'])
        }
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("✓ Training summary saved to training_summary.json")
    
    # ========================================================================
    # STEP 7: TEST INFERENCE PIPELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: TEST INFERENCE PIPELINE")
    print("=" * 80 + "\n")
    
    pipeline = InferencePipeline("diagnosis_model.pkl", "encoders.json")
    
    # Test cases
    test_cases = [
        {
            'name': 'Emergency Case - Possible Heart Attack',
            'symptoms': ['chest_pain', 'shortness_of_breath', 'sweating'],
            'severity': 'severe',
            'duration': '<1day'
        },
        {
            'name': 'Urgent Case - Possible Meningitis',
            'symptoms': ['high_fever', 'severe_headache', 'stiff_neck'],
            'severity': 'severe',
            'duration': '1-3days'
        },
        {
            'name': 'Non-Urgent Case - Common Cold',
            'symptoms': ['cough', 'runny_nose', 'fatigue'],
            'severity': 'mild',
            'duration': '1-3days'
        },
        {
            'name': 'Possible Flu',
            'symptoms': ['fever', 'body_aches', 'chills', 'fatigue'],
            'severity': 'moderate',
            'duration': '1-3days'
        }
    ]
    
    print("Running test predictions...\n")
    
    for test_case in test_cases:
        print("-" * 80)
        print(f"Test Case: {test_case['name']}")
        print("-" * 80)
        
        result = pipeline.predict(
            symptoms=test_case['symptoms'],
            severity=test_case['severity'],
            duration=test_case['duration'],
            top_k=3
        )
        
        print(pipeline.format_result_for_display(result))
        print("\n")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Files:")
    print("  ✓ medical_data.csv - Training dataset")
    print("  ✓ encoders.json - Feature encoders")
    print("  ✓ diagnosis_model.pkl - Trained model")
    print("  ✓ training_summary.json - Training metrics")
    print("  ✓ confusion_matrix.png - Confusion matrix visualization")
    print("  ✓ feature_importance.png - Feature importance plot")
    print("\nNext Steps:")
    print("  1. Review training_summary.json for model performance")
    print("  2. Check confusion_matrix.png to identify misclassifications")
    print("  3. Use inference_pipeline.py for making predictions")
    print("  4. Integrate with web frontend (see web_app.py)")
    print("=" * 80 + "\n")


def demo_inference_only():
    """
    Demo script for inference with pre-trained model
    Use this after running the complete pipeline at least once
    """
    print("=" * 80)
    print("INFERENCE DEMO (requires pre-trained model)")
    print("=" * 80 + "\n")
    
    try:
        pipeline = InferencePipeline("diagnosis_model.pkl", "encoders.json")
        
        # Interactive demo
        print("Enter symptoms (comma-separated), or 'quit' to exit:")
        print("Example: fever,cough,fatigue\n")
        
        while True:
            user_input = input("Symptoms: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            symptoms = [s.strip() for s in user_input.split(',')]
            
            # Optional: ask for severity and duration
            severity = input("Severity (mild/moderate/severe, or press Enter to skip): ").strip()
            duration = input("Duration (<1day/1-3days/3-7days/>7days, or press Enter to skip): ").strip()
            
            severity = severity if severity else None
            duration = duration if duration else None
            
            # Make prediction
            result = pipeline.predict(symptoms, severity, duration)
            
            print("\n" + pipeline.format_result_for_display(result))
            print("\n")
    
    except FileNotFoundError:
        print("⚠ Error: Model files not found!")
        print("Please run the complete pipeline first:")
        print("  python main_pipeline.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Run inference demo only
        demo_inference_only()
    else:
        # Run complete training pipeline
        run_complete_pipeline(
            use_sample_data=True,
            balance_data=True
        )
