from flask import Flask, request, jsonify
import pandas as pd
import json
import numpy as np
from datetime import datetime
from fraud_detection_pipeline import FraudDetectionPipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
pipeline = FraudDetectionPipeline(model_save_path="random_forest_model.pkl")
pipeline.load_model()

# Helper function to convert numpy values to JSON serializable format
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# API endpoint for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Fraud detection service is running'})

# API endpoint for batch predictions
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if it's a CSV file
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read the file
        transactions_df = pd.read_csv(file)
        
        # Make predictions
        predictions, probabilities = pipeline.predict_transaction(transactions_df)
        
        # Prepare results
        results = {
            'total_transactions': len(transactions_df),
            'suspicious_count': int(sum(predictions)),
            'genuine_count': int(len(predictions) - sum(predictions)),
            'predictions': [
                {
                    'index': i,
                    'transaction_id': str(transactions_df.iloc[i].get('MTN', f"TX-{i}")),
                    'prediction': 'Suspicious' if pred == 1 else 'Genuine',
                    'probability': float(prob),
                    'risk_level': 'High' if prob > 0.75 else 'Medium' if prob > 0.5 else 'Low'
                }
                for i, (pred, prob) in enumerate(zip(predictions, probabilities))
            ]
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for single transaction prediction
@app.route('/predict/transaction', methods=['POST'])
def predict_transaction():
    try:
        # Get JSON data
        transaction_data = request.get_json()
        
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Convert to DataFrame (single row)
        transaction_df = pd.DataFrame([transaction_data])
        
        # Make prediction
        predictions, probabilities = pipeline.predict_transaction(transaction_df)
        
        # Prepare result
        result = {
            'transaction_id': str(transaction_data.get('MTN', 'Unknown')),
            'prediction': 'Suspicious' if predictions[0] == 1 else 'Genuine',
            'probability': float(probabilities[0]),
            'risk_level': 'High' if probabilities[0] > 0.75 else 'Medium' if probabilities[0] > 0.5 else 'Low',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint to get all transactions with pagination
@app.route('/transactions/all_page', methods=['GET'])
def get_all_transactions_paginated():
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        # Load transactions (this is a placeholder - you would load from a database in a real app)
        # For demonstration, we'll create sample data
        total_transactions = 1174279  # Total count from the notebook
        
        # Generate sample transactions (in a real app, fetch from DB)
        sample_transactions = []
        for i in range((page-1)*per_page, min(page*per_page, total_transactions)):
            sample_transactions.append({
                'id': i + 3449,  # Starting ID from notebook
                'mtn': f"{100000000 + i}",
                'sender_id': f"{400000 + (i % 1000)}",
                'sender_legal_name': f"Company {i % 100}",
                'sender_country': 'UNITED STATES',
                'beneficiary_name': f"Beneficiary {i % 500}",
                'beneficiary_country': 'MEXICO',
                'total_sale': round(100 + (i % 1000), 2),
                'status': 'Paid Out',
                'sender_status_detail': 'Genuine' if i % 10 != 0 else 'Suspicious'
            })
        
        # Return paginated results
        return jsonify({
            'page': page,
            'per_page': per_page,
            'total': total_transactions,
            'total_pages': (total_transactions // per_page) + (1 if total_transactions % per_page > 0 else 0),
            'transactions': sample_transactions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint to stream all transactions
@app.route('/transactions/all_stream', methods=['GET'])
def stream_all_transactions():
    def generate():
        # This is a placeholder - in a real app, stream from DB with a cursor
        total_transactions = 1000  # Limit for streaming example
        
        yield '['  # Start of JSON array
        
        for i in range(total_transactions):
            transaction = {
                'id': i + 1,
                'mtn': f"{100000000 + i}",
                'sender_id': f"{400000 + (i % 1000)}",
                'sender_legal_name': f"Company {i % 100}",
                'total_sale': round(100 + (i % 1000), 2),
                'status': 'Paid Out',
                'sender_status_detail': 'Genuine' if i % 10 != 0 else 'Suspicious'
            }
            
            # Add comma for all but the last item
            if i < total_transactions - 1:
                yield json.dumps(transaction, default=convert_to_serializable) + ','
            else:
                yield json.dumps(transaction, default=convert_to_serializable)
        
        yield ']'  # End of JSON array
    
    return app.response_class(generate(), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True) 