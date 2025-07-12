#!/usr/bin/env python3
"""
Fraud Detection Service for Walmart Hack Backend
Uses RGTAN model by default for fraud detection
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import logging


# Add the current directory to the path
sys.path.append('.')

# Import the antifraud components
from app import load_models, predict_fraud

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables to store loaded models
models_loaded = False

def initialize_models():
    """Initialize the fraud detection models"""
    global models_loaded
    try:
        # Stay in current directory (models are here now)
        # os.chdir('../untitled folder')
        
        # Load the models
        load_models()
        models_loaded = True
        logger.info("✅ Fraud detection models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

def transform_payment_to_transaction(payment_data):
    """Transform payment data to transaction format expected by the model"""
    try:
        # Handle direct transaction format from Node.js
        if 'Time' in payment_data and 'Source' in payment_data:
            # Direct transaction format - just convert to DataFrame
            transaction_data = pd.DataFrame({
                'Time': [payment_data.get('Time')],
                'Source': [payment_data.get('Source')],
                'Target': [payment_data.get('Target', 'Walmart India')],
                'Amount': [float(payment_data.get('Amount', 0))],
                'Location': [payment_data.get('Location', 'India')],
                'Type': [payment_data.get('Type', 'card')]
            })
        else:
            # Legacy format - transform payment data
            transaction_data = pd.DataFrame({
                'Time': [int(datetime.now().timestamp())],
                'Source': [payment_data.get('userId', 'unknown')],
                'Target': [payment_data.get('merchantId', 'Walmart India')],
                'Amount': [float(payment_data.get('amount', 0))],
                'Location': [payment_data.get('location', 'unknown')],
                'Type': [payment_data.get('method', 'card')]
            })
        
        return transaction_data
    except Exception as e:
        logger.error(f"Error transforming payment data: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_fraud_endpoint():
    """Predict fraud for a payment transaction"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'status': 'error'
            }), 500
            
        # Get payment data from request
        payment_data = request.json
        
        if not payment_data:
            return jsonify({
                'error': 'No payment data provided',
                'status': 'error'
            }), 400
        
        # Transform payment data to transaction format
        transaction_data = transform_payment_to_transaction(payment_data)
        
        if transaction_data is None:
            return jsonify({
                'error': 'Failed to transform payment data',
                'status': 'error'
            }), 400
        
        # Make prediction using RGTAN model
        fraud_probs, pred_labels = predict_fraud(transaction_data, 'rgtan')
        
        # Convert numpy arrays to lists for JSON serialization
        if fraud_probs is not None:
            fraud_probs = fraud_probs.tolist()
        if pred_labels is not None:
            pred_labels = pred_labels.tolist()

        # Get the fraud probability and prediction
        fraud_probability = float(fraud_probs[0]) if fraud_probs else 0.0
        prediction = int(pred_labels[0]) if pred_labels else 0
        
        # Determine risk level
        if fraud_probability > 0.7:
            risk_level = 'HIGH'
        elif fraud_probability > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Return prediction results
        return jsonify({
            'status': 'success',
            'fraud_probability': fraud_probability,
            'prediction': prediction,
            'risk_level': risk_level,
            'is_fraud': prediction == 1,
            'model_used': 'rgtan',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in fraud prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_fraud_batch():
    """Predict fraud for multiple transactions"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'status': 'error'
            }), 500
            
        # Get batch of payment data
        batch_data = request.json
        
        if not batch_data or 'payments' not in batch_data:
            return jsonify({
                'error': 'No payment data provided',
                'status': 'error'
            }), 400
        
        payments = batch_data['payments']
        results = []
        
        for payment_data in payments:
            # Transform payment data to transaction format
            transaction_data = transform_payment_to_transaction(payment_data)
            
            if transaction_data is None:
                continue
            
            # Make prediction using RGTAN model
            fraud_probs, pred_labels = predict_fraud(transaction_data, 'rgtan')
            
            # Get the fraud probability and prediction
            fraud_probability = float(fraud_probs[0]) if fraud_probs is not None else 0.0
            prediction = int(pred_labels[0]) if pred_labels is not None else 0
            
            # Determine risk level
            if fraud_probability > 0.7:
                risk_level = 'HIGH'
            elif fraud_probability > 0.3:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            results.append({
                'payment_id': payment_data.get('_id'),
                'fraud_probability': fraud_probability,
                'prediction': prediction,
                'risk_level': risk_level,
                'is_fraud': prediction == 1
            })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'model_used': 'rgtan',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch fraud prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Initialize models on startup
    if initialize_models():
        logger.info("🚀 Starting fraud detection service...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        logger.error("❌ Failed to initialize models. Exiting.")
        sys.exit(1)
