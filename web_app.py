"""
Web Application
Flask-based web interface for the diagnosis prediction bot
"""

from flask import Flask, render_template, request, jsonify
from inference_pipeline import InferencePipeline
import os

app = Flask(__name__)

# Initialize pipeline
try:
    pipeline = InferencePipeline("diagnosis_model.pkl", "encoders.json")
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("⚠ Warning: Model not loaded. Run main_pipeline.py first.")


@app.route('/')
def home():
    """Main page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        symptoms = data.get('symptoms', [])
        severity = data.get('severity')
        duration = data.get('duration')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Make prediction
        result = pipeline.predict(symptoms, severity, duration, top_k=3)
        
        # Format response
        response = {
            'timestamp': result.timestamp,
            'safety_check': {
                'urgency': result.safety_check['urgency'].value,
                'message': result.safety_check['message'],
                'explanation': result.safety_check['explanation'],
                'flags': result.safety_check['flags']
            },
            'predictions': [
                {'condition': cond, 'probability': prob}
                for cond, prob in result.top_predictions
            ],
            'confidence': result.confidence_level,
            'recommendations': result.recommendations,
            'disclaimers': result.disclaimers
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get list of available symptoms"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    symptoms = pipeline.preprocessor.symptom_vocabulary.tolist()
    return jsonify({'symptoms': sorted(symptoms)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
