"""
Inference Pipeline
Orchestrates the complete diagnosis prediction workflow
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from data_preprocessing import DataPreprocessor
from model_training import DiagnosisModel
from safety_module import SafetyChecker, UrgencyLevel, DisclaimerGenerator


@dataclass
class PredictionResult:
    """Structured prediction result"""
    top_predictions: List[Tuple[str, float]]
    safety_check: Dict
    confidence_level: str
    timestamp: str
    user_symptoms: List[str]
    severity: Optional[str]
    duration: Optional[str]
    recommendations: str
    disclaimers: List[str]


class InferencePipeline:
    """Complete inference pipeline for diagnosis prediction"""
    
    def __init__(self, model_path: str = "diagnosis_model.pkl", 
                 encoders_path: str = "encoders.json"):
        """
        Initialize the inference pipeline
        
        Args:
            model_path: Path to trained model file
            encoders_path: Path to encoders file
        """
        self.preprocessor = DataPreprocessor()
        self.model = DiagnosisModel()
        self.safety_checker = SafetyChecker()
        self.disclaimer_gen = DisclaimerGenerator()
        
        # Load model and encoders
        self._load_components(model_path, encoders_path)
        
    def _load_components(self, model_path: str, encoders_path: str):
        """Load model and encoders"""
        try:
            self.model.load_model(model_path)
            self.preprocessor.load_encoders(encoders_path)
            print("✓ Inference pipeline initialized successfully")
        except FileNotFoundError as e:
            print(f"⚠ Warning: Could not load components: {e}")
            print("  Pipeline will need trained components before use")
    
    def predict(self, symptoms: List[str], 
                severity: Optional[str] = None,
                duration: Optional[str] = None,
                top_k: int = 3) -> PredictionResult:
        """
        Complete prediction pipeline with safety checks
        
        Args:
            symptoms: List of symptom strings
            severity: Optional severity level ('mild', 'moderate', 'severe')
            duration: Optional duration ('<1day', '1-3days', '3-7days', '>7days')
            top_k: Number of top predictions to return
            
        Returns:
            PredictionResult object with all prediction information
        """
        # Step 1: Safety check (runs first - most important)
        safety_result = self.safety_checker.check_symptoms(symptoms)
        
        # Step 2: Encode symptoms for model input
        feature_vector = self.preprocessor.encode_user_symptoms(
            symptoms, severity, duration
        )
        
        # Step 3: Get model predictions
        predictions = self.model.predict_top_k(
            feature_vector, 
            k=top_k, 
            class_names=self.model.class_names
        )[0]  # Get first (and only) result
        
        # Step 4: Determine confidence level
        confidence = self._assess_confidence(predictions)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            safety_result['urgency'],
            predictions,
            symptoms
        )
        
        # Step 6: Compile disclaimers
        disclaimers = [
            self.disclaimer_gen.get_main_disclaimer(),
            self.disclaimer_gen.get_ai_limitation_disclaimer()
        ]
        
        # Create structured result
        result = PredictionResult(
            top_predictions=predictions,
            safety_check=safety_result,
            confidence_level=confidence,
            timestamp=datetime.now().isoformat(),
            user_symptoms=symptoms,
            severity=severity,
            duration=duration,
            recommendations=recommendations,
            disclaimers=disclaimers
        )
        
        return result
    
    def _assess_confidence(self, predictions: List[Tuple[str, float]]) -> str:
        """
        Assess confidence level based on prediction probabilities
        
        Args:
            predictions: List of (condition, probability) tuples
            
        Returns:
            Confidence level string
        """
        if not predictions:
            return "VERY_LOW"
        
        top_prob = predictions[0][1]
        prob_gap = predictions[0][1] - predictions[1][1] if len(predictions) > 1 else top_prob
        
        # High confidence: top probability > 0.7 and significant gap
        if top_prob > 0.7 and prob_gap > 0.2:
            return "HIGH"
        
        # Medium confidence: top probability > 0.5 or moderate gap
        elif top_prob > 0.5 or prob_gap > 0.1:
            return "MEDIUM"
        
        # Low confidence: predictions are close together
        elif top_prob > 0.3:
            return "LOW"
        
        # Very low confidence
        else:
            return "VERY_LOW"
    
    def _generate_recommendations(self, urgency: UrgencyLevel, 
                                 predictions: List[Tuple[str, float]],
                                 symptoms: List[str]) -> str:
        """Generate contextual recommendations"""
        
        # Get triage recommendation
        triage = self.safety_checker.get_triage_recommendation(urgency, predictions)
        
        # Add prediction-specific advice
        if predictions and urgency != UrgencyLevel.EMERGENCY:
            top_condition = predictions[0][0]
            condition_advice = self._get_condition_specific_advice(top_condition)
            
            full_recommendation = f"{triage}\n\n{condition_advice}"
        else:
            full_recommendation = triage
        
        return full_recommendation
    
    def _get_condition_specific_advice(self, condition: str) -> str:
        """Get condition-specific self-care advice"""
        
        # Basic advice mapping (would be more comprehensive in production)
        advice_map = {
            'Common Cold': """
SELF-CARE FOR COMMON COLD:
• Get plenty of rest and stay hydrated
• Use over-the-counter pain relievers for discomfort
• Try saline nasal sprays or rinses
• Use a humidifier to ease congestion
• Gargle with warm salt water for sore throat
            """.strip(),
            
            'Flu': """
SELF-CARE FOR INFLUENZA:
• Rest and drink plenty of fluids
• Take fever-reducing medications if needed
• Stay home to avoid spreading illness
• Consider antiviral medications (consult doctor within 48 hours)
• Monitor for complications like difficulty breathing
            """.strip(),
            
            'Migraine': """
SELF-CARE FOR MIGRAINE:
• Rest in a quiet, dark room
• Apply cold or warm compress to head/neck
• Stay hydrated
• Avoid known triggers (certain foods, stress, bright lights)
• Consider over-the-counter pain relievers
• Track triggers and patterns for future prevention
            """.strip(),
            
            'Gastroenteritis': """
SELF-CARE FOR GASTROENTERITIS:
• Stay well hydrated (water, clear broths, oral rehydration solutions)
• Eat bland foods when able (BRAT diet: bananas, rice, applesauce, toast)
• Avoid dairy, caffeine, alcohol temporarily
• Rest frequently
• Practice good hand hygiene to prevent spread
            """.strip(),
        }
        
        return advice_map.get(condition, """
GENERAL SELF-CARE RECOMMENDATIONS:
• Monitor symptoms and note any changes
• Get adequate rest and hydration
• Follow up with healthcare provider as needed
• Seek immediate care if symptoms worsen rapidly
        """.strip())
    
    def format_result_for_display(self, result: PredictionResult) -> str:
        """
        Format prediction result as human-readable text
        
        Args:
            result: PredictionResult object
            
        Returns:
            Formatted string for display
        """
        output = []
        
        # Header
        output.append("=" * 70)
        output.append("DIAGNOSIS PREDICTION RESULTS")
        output.append("=" * 70)
        output.append(f"Timestamp: {result.timestamp}")
        output.append(f"Symptoms: {', '.join(result.user_symptoms)}")
        
        if result.severity:
            output.append(f"Severity: {result.severity}")
        if result.duration:
            output.append(f"Duration: {result.duration}")
        
        output.append("")
        
        # Safety Check Results
        output.append("-" * 70)
        output.append("SAFETY ASSESSMENT")
        output.append("-" * 70)
        output.append(result.safety_check['message'])
        output.append(f"\n{result.safety_check['explanation']}")
        output.append("")
        
        # If emergency, emphasize the warning
        if result.safety_check['urgency'] == UrgencyLevel.EMERGENCY:
            output.append("⚠️  " * 20)
            output.append("DO NOT DELAY - SEEK IMMEDIATE EMERGENCY CARE")
            output.append("⚠️  " * 20)
            output.append("")
        
        # Prediction Results
        output.append("-" * 70)
        output.append("AI PREDICTIONS")
        output.append("-" * 70)
        output.append(f"Confidence Level: {result.confidence_level}")
        output.append("")
        output.append("Top Predicted Conditions:")
        
        for i, (condition, probability) in enumerate(result.top_predictions, 1):
            bar_length = int(probability * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            output.append(f"{i}. {condition:30s} {probability:6.2%} [{bar}]")
        
        output.append("")
        
        # Recommendations
        output.append("-" * 70)
        output.append("RECOMMENDATIONS")
        output.append("-" * 70)
        output.append(result.recommendations)
        output.append("")
        
        # Disclaimers
        output.append("-" * 70)
        output.append("IMPORTANT DISCLAIMERS")
        output.append("-" * 70)
        for disclaimer in result.disclaimers:
            output.append(disclaimer)
            output.append("")
        
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def format_result_as_json(self, result: PredictionResult) -> str:
        """
        Format prediction result as JSON
        
        Args:
            result: PredictionResult object
            
        Returns:
            JSON string
        """
        data = {
            'timestamp': result.timestamp,
            'input': {
                'symptoms': result.user_symptoms,
                'severity': result.severity,
                'duration': result.duration
            },
            'safety_check': {
                'urgency_level': result.safety_check['urgency'].value,
                'message': result.safety_check['message'],
                'explanation': result.safety_check['explanation'],
                'flags': result.safety_check['flags'],
                'action': result.safety_check['action']
            },
            'predictions': [
                {'condition': cond, 'probability': float(prob)}
                for cond, prob in result.top_predictions
            ],
            'confidence_level': result.confidence_level,
            'recommendations': result.recommendations,
            'disclaimers': result.disclaimers
        }
        
        return json.dumps(data, indent=2)


class BatchInference:
    """Handle batch inference for multiple cases"""
    
    def __init__(self, pipeline: InferencePipeline):
        self.pipeline = pipeline
    
    def predict_batch(self, cases: List[Dict]) -> List[PredictionResult]:
        """
        Process multiple cases
        
        Args:
            cases: List of dictionaries with 'symptoms', 'severity', 'duration'
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for i, case in enumerate(cases, 1):
            print(f"Processing case {i}/{len(cases)}...")
            
            result = self.pipeline.predict(
                symptoms=case.get('symptoms', []),
                severity=case.get('severity'),
                duration=case.get('duration')
            )
            
            results.append(result)
        
        print(f"✓ Batch inference complete: {len(results)} cases processed")
        return results
    
    def generate_batch_report(self, results: List[PredictionResult], 
                            output_path: str = "batch_report.json"):
        """Generate summary report for batch predictions"""
        
        report = {
            'total_cases': len(results),
            'timestamp': datetime.now().isoformat(),
            'urgency_distribution': {},
            'confidence_distribution': {},
            'most_common_predictions': {},
            'cases': []
        }
        
        # Aggregate statistics
        for result in results:
            # Count urgency levels
            urgency = result.safety_check['urgency'].value
            report['urgency_distribution'][urgency] = \
                report['urgency_distribution'].get(urgency, 0) + 1
            
            # Count confidence levels
            report['confidence_distribution'][result.confidence_level] = \
                report['confidence_distribution'].get(result.confidence_level, 0) + 1
            
            # Count top predictions
            top_condition = result.top_predictions[0][0]
            report['most_common_predictions'][top_condition] = \
                report['most_common_predictions'].get(top_condition, 0) + 1
            
            # Add case summary
            report['cases'].append({
                'symptoms': result.user_symptoms,
                'top_prediction': result.top_predictions[0][0],
                'urgency': urgency,
                'confidence': result.confidence_level
            })
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Batch report saved to {output_path}")
        return report


if __name__ == "__main__":
    # Demo usage
    print("=== Inference Pipeline Demo ===\n")
    
    # Note: This demo won't work without trained model and encoders
    # See main_pipeline.py for complete training + inference example
    
    test_cases = [
        {
            'name': 'Emergency Case',
            'symptoms': ['chest_pain', 'shortness_of_breath', 'sweating'],
            'severity': 'severe',
            'duration': '<1day'
        },
        {
            'name': 'Urgent Case',
            'symptoms': ['high_fever', 'severe_headache', 'stiff_neck'],
            'severity': 'severe',
            'duration': '1-3days'
        },
        {
            'name': 'Common Illness',
            'symptoms': ['cough', 'runny_nose', 'fatigue'],
            'severity': 'mild',
            'duration': '1-3days'
        }
    ]
    
    print("Test cases defined:")
    for case in test_cases:
        print(f"  - {case['name']}: {', '.join(case['symptoms'])}")
    
    print("\n⚠ To run inference, train the model first using main_pipeline.py")
