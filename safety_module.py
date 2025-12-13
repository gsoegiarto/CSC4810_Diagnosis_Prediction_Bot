"""
Safety Module
Red-flag detection for urgent/emergency symptoms requiring immediate medical attention
"""

from typing import List, Dict, Tuple
from enum import Enum


class UrgencyLevel(Enum):
    """Classification of medical urgency"""
    EMERGENCY = "EMERGENCY"  # Call 911 immediately
    URGENT = "URGENT"  # Seek medical care within hours
    NON_URGENT = "NON_URGENT"  # Can wait for regular appointment
    NORMAL = "NORMAL"  # Self-care or routine monitoring


class SafetyChecker:
    """Detects red-flag symptoms that require immediate medical attention"""
    
    def __init__(self):
        # Emergency symptoms (life-threatening)
        self.emergency_symptoms = {
            'chest_pain', 'severe_chest_pain', 'crushing_chest_pain',
            'difficulty_breathing', 'shortness_of_breath', 'severe_shortness_of_breath',
            'unable_to_breathe',
            'loss_of_consciousness', 'unconscious', 'unresponsive',
            'severe_bleeding', 'uncontrolled_bleeding', 'hemorrhage',
            'stroke_symptoms', 'facial_drooping', 'arm_weakness', 'speech_difficulty',
            'severe_head_injury', 'head_trauma',
            'seizure', 'convulsions',
            'severe_allergic_reaction', 'anaphylaxis', 'throat_swelling',
            'coughing_blood', 'vomiting_blood', 'blood_in_stool',
            'severe_burns',
            'poisoning', 'overdose',
            'suicidal_thoughts', 'thoughts_of_self_harm',
            'severe_abdominal_pain',
            'sudden_confusion', 'disorientation',
            'blue_lips', 'blue_fingernails', 'cyanosis',
        }
        
        # Urgent symptoms (need medical attention within hours)
        self.urgent_symptoms = {
            'high_fever', 'fever_over_103', 'persistent_fever',
            'severe_headache', 'worst_headache_of_life',
            'stiff_neck', 'neck_stiffness',
            'persistent_vomiting', 'severe_vomiting',
            'severe_diarrhea', 'bloody_diarrhea',
            'severe_pain', 'unbearable_pain',
            'eye_pain', 'vision_loss', 'sudden_vision_changes',
            'severe_weakness', 'paralysis',
            'difficulty_swallowing',
            'severe_rash', 'rapidly_spreading_rash',
            'persistent_cough_with_blood',
            'severe_dehydration',
            'confusion', 'altered_mental_state',
            'severe_anxiety', 'panic_attack',
            'irregular_heartbeat', 'heart_palpitations', 'rapid_heartbeat',
        }
        
        # Dangerous symptom combinations
        self.dangerous_combinations = [
            (['chest_pain', 'sweating', 'shortness_of_breath'], 'Possible heart attack'),
            (['severe_headache', 'stiff_neck', 'fever'], 'Possible meningitis'),
            (['abdominal_pain', 'fever', 'vomiting'], 'Possible appendicitis'),
            (['facial_drooping', 'arm_weakness', 'speech_difficulty'], 'Possible stroke'),
            (['wheezing', 'shortness_of_breath', 'chest_tightness'], 'Possible severe asthma'),
        ]
        
    def normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text for matching"""
        return symptom.lower().strip().replace(' ', '_')
    
    def check_symptoms(self, symptoms: List[str]) -> Dict:
        """
        Check symptoms for red flags and determine urgency level
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Dictionary with urgency info and recommendations
        """
        normalized_symptoms = [self.normalize_symptom(s) for s in symptoms]
        
        # Check for emergency symptoms
        emergency_flags = [s for s in normalized_symptoms if s in self.emergency_symptoms]
        if emergency_flags:
            return {
                'urgency': UrgencyLevel.EMERGENCY,
                'flags': emergency_flags,
                'message': '🚨 EMERGENCY: These symptoms require immediate medical attention. Call 911 or go to the nearest emergency room NOW.',
                'action': 'CALL_911',
                'explanation': self._get_emergency_explanation(emergency_flags)
            }
        
        # Check for dangerous combinations
        for combo, condition in self.dangerous_combinations:
            normalized_combo = [self.normalize_symptom(s) for s in combo]
            if all(symptom in normalized_symptoms for symptom in normalized_combo):
                return {
                    'urgency': UrgencyLevel.EMERGENCY,
                    'flags': combo,
                    'message': f'🚨 EMERGENCY: {condition}. Seek immediate medical attention. Call 911 or go to the nearest emergency room.',
                    'action': 'CALL_911',
                    'explanation': f'The combination of these symptoms may indicate {condition.lower()}, which requires immediate evaluation.'
                }
        
        # Check for urgent symptoms
        urgent_flags = [s for s in normalized_symptoms if s in self.urgent_symptoms]
        if urgent_flags:
            return {
                'urgency': UrgencyLevel.URGENT,
                'flags': urgent_flags,
                'message': '⚠️ URGENT: These symptoms require prompt medical evaluation. Contact your doctor or visit urgent care within the next few hours.',
                'action': 'SEEK_CARE_SOON',
                'explanation': self._get_urgent_explanation(urgent_flags)
            }
        
        # No red flags detected
        return {
            'urgency': UrgencyLevel.NON_URGENT,
            'flags': [],
            'message': 'ℹ️ No immediate red flags detected. However, this is not a substitute for professional medical advice.',
            'action': 'MONITOR_OR_SCHEDULE',
            'explanation': 'Based on the symptoms provided, there are no immediate emergency indicators. Consider scheduling a regular appointment with your healthcare provider if symptoms persist or worsen.'
        }
    
    def _get_emergency_explanation(self, flags: List[str]) -> str:
        """Get detailed explanation for emergency flags"""
        explanations = {
            'chest_pain': 'Chest pain can indicate a heart attack or other serious cardiac condition.',
            'difficulty_breathing': 'Severe breathing difficulty can indicate respiratory failure or other life-threatening conditions.',
            'loss_of_consciousness': 'Loss of consciousness requires immediate evaluation for serious underlying causes.',
            'severe_bleeding': 'Uncontrolled bleeding is a medical emergency that requires immediate intervention.',
            'seizure': 'Seizures require immediate medical evaluation to prevent complications.',
            'stroke_symptoms': 'Stroke symptoms require immediate treatment - every minute matters.',
            'anaphylaxis': 'Severe allergic reactions can be life-threatening and require immediate treatment.',
        }
        
        details = []
        for flag in flags[:3]:  # Limit to top 3 for clarity
            if flag in explanations:
                details.append(explanations[flag])
        
        if not details:
            return 'These symptoms may indicate a serious medical emergency.'
        
        return ' '.join(details)
    
    def _get_urgent_explanation(self, flags: List[str]) -> str:
        """Get detailed explanation for urgent flags"""
        explanations = {
            'high_fever': 'Persistent high fever can indicate serious infection.',
            'severe_headache': 'Severe or unusual headaches should be evaluated promptly.',
            'stiff_neck': 'Neck stiffness with other symptoms may indicate meningitis.',
            'persistent_vomiting': 'Persistent vomiting can lead to dehydration and may indicate serious conditions.',
            'severe_pain': 'Severe pain requires medical evaluation to identify the cause.',
        }
        
        details = []
        for flag in flags[:3]:
            if flag in explanations:
                details.append(explanations[flag])
        
        if not details:
            return 'These symptoms warrant prompt medical evaluation.'
        
        return ' '.join(details)
    
    def get_triage_recommendation(self, urgency: UrgencyLevel, 
                                  predicted_conditions: List[Tuple[str, float]]) -> str:
        """
        Generate triage recommendation based on urgency and predictions
        
        Args:
            urgency: UrgencyLevel enum
            predicted_conditions: List of (condition, probability) tuples
            
        Returns:
            Formatted recommendation string
        """
        if urgency == UrgencyLevel.EMERGENCY:
            return """
🚨 IMMEDIATE ACTION REQUIRED:
• Call 911 or emergency services NOW
• Do not drive yourself - call for emergency transport
• If alone, call a neighbor or family member for help
• Stay calm and follow emergency dispatcher instructions
            """.strip()
        
        elif urgency == UrgencyLevel.URGENT:
            return """
⚠️ PROMPT MEDICAL ATTENTION NEEDED:
• Contact your doctor's office immediately
• Visit an urgent care clinic within the next few hours
• If after hours, consider an emergency room visit
• Do not delay - symptoms require timely evaluation
            """.strip()
        
        elif urgency == UrgencyLevel.NON_URGENT:
            top_condition = predicted_conditions[0][0] if predicted_conditions else "your condition"
            return f"""
ℹ️ RECOMMENDED NEXT STEPS:
• Schedule an appointment with your primary care doctor
• Monitor your symptoms - note any changes or worsening
• Consider {top_condition} based on AI prediction
• Seek care if symptoms worsen or new symptoms appear
            """.strip()
        
        else:  # NORMAL
            return """
✓ SELF-CARE AND MONITORING:
• Continue monitoring your symptoms
• Practice good self-care (rest, hydration, nutrition)
• Consider over-the-counter remedies if appropriate
• Schedule a routine checkup if symptoms persist beyond a few days
            """.strip()


class DisclaimerGenerator:
    """Generates appropriate medical disclaimers"""
    
    @staticmethod
    def get_main_disclaimer() -> str:
        """Get the main medical disclaimer"""
        return """
⚠️ IMPORTANT MEDICAL DISCLAIMER:
This tool is for informational purposes only and is NOT a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
with any questions you may have regarding a medical condition. Never disregard professional medical 
advice or delay seeking it because of information from this tool. In case of emergency, call 911 
or your local emergency number immediately.
        """.strip()
    
    @staticmethod
    def get_ai_limitation_disclaimer() -> str:
        """Disclaimer about AI limitations"""
        return """
🤖 AI SYSTEM LIMITATIONS:
This system uses machine learning to predict potential conditions based on reported symptoms. It has 
limitations and may not account for your complete medical history, current medications, or other 
important factors. The predictions should be considered as possible conditions to discuss with your 
healthcare provider, not definitive diagnoses.
        """.strip()
    
    @staticmethod
    def get_emergency_disclaimer() -> str:
        """Disclaimer for emergency situations"""
        return """
🚨 IF YOU'RE EXPERIENCING A MEDICAL EMERGENCY:
Do not use this tool. Call 911 or your local emergency number immediately. Signs of emergency include:
• Chest pain or pressure
• Difficulty breathing
• Severe bleeding
• Loss of consciousness
• Stroke symptoms (facial drooping, arm weakness, speech difficulty)
• Severe allergic reactions
        """.strip()


if __name__ == "__main__":
    # Demo usage
    print("=== Safety Module Demo ===\n")
    
    checker = SafetyChecker()
    disclaimer = DisclaimerGenerator()
    
    # Test cases
    test_cases = [
        {
            'name': 'Emergency Case',
            'symptoms': ['chest pain', 'shortness of breath', 'sweating']
        },
        {
            'name': 'Urgent Case',
            'symptoms': ['high fever', 'severe headache', 'stiff neck']
        },
        {
            'name': 'Non-Urgent Case',
            'symptoms': ['cough', 'runny nose', 'fatigue']
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"Symptoms: {', '.join(test['symptoms'])}")
        print('='*60)
        
        result = checker.check_symptoms(test['symptoms'])
        
        print(f"\nUrgency Level: {result['urgency'].value}")
        print(f"\n{result['message']}")
        print(f"\n{result['explanation']}")
        
        if result['flags']:
            print(f"\nRed Flags Detected: {', '.join(result['flags'])}")
        
        print(f"\n{checker.get_triage_recommendation(result['urgency'], [('Common Cold', 0.8)])}")
    
    print(f"\n\n{'='*60}")
    print("DISCLAIMERS")
    print('='*60)
    print(f"\n{disclaimer.get_main_disclaimer()}")
    print(f"\n{disclaimer.get_ai_limitation_disclaimer()}")
