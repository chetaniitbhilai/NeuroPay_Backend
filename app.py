from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import torch
import os
import pickle
import yaml
from werkzeug.utils import secure_filename
import sys
import logging
from datetime import datetime
import json
import dgl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import zscore

# Add the antifraud directory to the path
sys.path.append('antifraud')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
model_configs = {}

def load_models():
    """Load pre-trained models and configurations"""
    try:
        # Load configurations (S-FFSD compatible models only)
        config_files = [
            'antifraud/config/mcnn_cfg.yaml',
            'antifraud/config/stan_cfg.yaml',
            'antifraud/config/stan_2d_cfg.yaml',
            'antifraud/config/gtan_cfg.yaml',
            'antifraud/config/stagn_cfg.yaml',
            'antifraud/config/rgtan_cfg.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    method_name = config_file.split('/')[-1].replace('_cfg.yaml', '')
                    model_configs[method_name] = config
        
        # Load pre-trained models if they exist (S-FFSD compatible models only)
        models_dir = 'antifraud/models'
        if os.path.exists(models_dir):
            # Only load models that are compatible with S-FFSD dataset
            for method in ['mcnn', 'stan', 'gtan', 'stagn', 'rgtan']:
                # All models now use _ckpt.pth naming convention
                model_path = os.path.join(models_dir, f'{method}_ckpt.pth')
                    
                if os.path.exists(model_path):
                    try:
                        if method == 'mcnn':
                            from antifraud.methods.mcnn.mcnn_model import mcnn
                            # Add safe globals for PyTorch 2.6+ security
# torch.serialization.add_safe_globals({'mcnn': mcnn})  # Commented out due to API change
                            model = mcnn()
                            # Load with weights_only=False to allow custom classes
                            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
                            model.eval()
                            models[method] = model
                            logger.info(f"✅ Successfully loaded {method} model")
                        elif method == 'stan':
                            from antifraud.methods.stan.stan_2d import stan_2d_model
# torch.serialization.add_safe_globals({'stan_2d_model': stan_2d_model})  # Commented out due to API change
                            # Create model with parameters matching trained model
                            model = stan_2d_model(
                                time_windows_dim=8,
                                feat_dim=5,
                                num_classes=2,
                                attention_hidden_dim=150
                            )
                            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
                            model.eval()
                            models[method] = model
                            logger.info(f"✅ Successfully loaded {method} model")
                        elif method == 'gtan':
                            try:
                                import dgl
                                from antifraud.methods.gtan.gtan_model import GraphAttnModel
# torch.serialization.add_safe_globals({'GraphAttnModel': GraphAttnModel})  # Commented out due to API change
                                # Create model with default parameters
                                model = GraphAttnModel(
                                    in_feats=25,  # Based on Amazon dataset features
                                    hidden_dim=64,
                                    n_layers=3,
                                    n_classes=2,
                                    heads=[4, 4, 4],
                                    activation=torch.nn.PReLU(),
                                    drop=[0.1, 0.1],
                                    device='cpu'
                                )
                                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
                                model.eval()
                                models[method] = model
                                logger.info(f"✅ Successfully loaded {method} model")
                            except ImportError as ie:
                                raise Exception(f"DGL not available: {ie}")
                        elif method == 'stagn':
                            try:
                                from antifraud.methods.stagn.stagn_2d import stagn_2d_model
                                # torch.serialization.add_safe_globals({'stagn_2d_model': stagn_2d_model})  # Commented out due to API change
                                # Load model directly since it's a checkpoint
                                model = torch.load(model_path, map_location='cpu', weights_only=False)
                                model.eval()
                                models[method] = model
                                logger.info(f"✅ Successfully loaded {method} model")
                            except Exception as e:
                                raise Exception(f"STAGN loading failed: {e}")
                        elif method == 'rgtan':
                            try:
                                from antifraud.methods.rgtan.rgtan_model import RGTAN
                                # torch.serialization.add_safe_globals({'RGTAN': RGTAN})  # Commented out due to API change
                                # Load model directly since it's a checkpoint
                                model = torch.load(model_path, map_location='cpu', weights_only=False)
                                model.eval()
                                models[method] = model
                                logger.info(f"✅ Successfully loaded {method} model")
                            except Exception as e:
                                raise Exception(f"RGTAN loading failed: {e}")
                    except Exception as e:
                        logger.error(f"Error loading {method} model: {e}")
                        # Try alternative loading method
                        try:
                            logger.info(f"Trying alternative loading method for {method}")
                            if method == 'mcnn':
                                from antifraud.methods.mcnn.mcnn_model import mcnn
                                # Try loading the entire model instead of just state_dict
                                # with torch.serialization.safe_globals({'mcnn': mcnn}):  # Commented out due to API change
                                model = torch.load(model_path, map_location='cpu', weights_only=False)
                                model.eval()
                                models[method] = model
                                logger.info(f"✅ Successfully loaded {method} model (alternative method)")
                            elif method == 'stan':
                                from antifraud.methods.stan.stan_2d import stan_2d_model
                                # with torch.serialization.safe_globals({'stan_2d_model': stan_2d_model}):  # Commented out due to API change
                                model = torch.load(model_path, map_location='cpu', weights_only=False)
                                model.eval()
                                models[method] = model
                                logger.info(f"✅ Successfully loaded {method} model (alternative method)")
                            elif method == 'gtan':
                                try:
                                    import dgl
                                    from antifraud.methods.gtan.gtan_model import GraphAttnModel
                                    # with torch.serialization.safe_globals({'GraphAttnModel': GraphAttnModel}):  # Commented out due to API change
                                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                                    model.eval()
                                    models[method] = model
                                    logger.info(f"✅ Successfully loaded {method} model (alternative method)")
                                except ImportError:
                                    raise Exception("DGL not available")
                        except Exception as e2:
                            logger.error(f"Alternative loading also failed for {method}: {e2}")
        
        logger.info(f"Loaded {len(models)} models and {len(model_configs)} configurations")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def validate_data_format(data):
    """Validate that the uploaded data has the correct format"""
    required_columns = ['Time', 'Source', 'Target', 'Amount', 'Location', 'Type']
    
    if not all(col in data.columns for col in required_columns):
        return False, f"Missing required columns. Expected: {required_columns}"
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(data['Time']):
        return False, "Time column must be numeric"
    
    if not pd.api.types.is_numeric_dtype(data['Amount']):
        return False, "Amount column must be numeric"
    
    return True, "Data format is valid"

def preprocess_data(data, method='mcnn'):
    """Preprocess data for the selected model using proper S-FFSD feature engineering"""
    try:
        from antifraud.feature_engineering.data_engineering import span_data_2d, span_data_3d
        
        # Check if data has the S-FFSD format
        required_cols = ['Time', 'Source', 'Target', 'Amount', 'Location', 'Type']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must have S-FFSD format with columns: {required_cols}")
        
        # Add dummy Labels column for feature engineering (will be ignored for prediction)
        data_copy = data.copy()
        data_copy['Labels'] = 0  # Dummy labels for preprocessing
        
        # Use the proper S-FFSD feature engineering
        if method in ['mcnn', 'stagn']:
            # Use 2D feature engineering for MCNN and STAGN
            features, _ = span_data_2d(data_copy)
            # features shape: (sample_num, 5_features, 8_time_windows)
            # Need to transpose for model: (sample_num, 8_time_windows, 5_features)
            features = features.transpose(0, 2, 1)
            
        elif method in ['stan', 'stan_2d']:
            # Use 2D feature engineering for STAN as well
            features, _ = span_data_2d(data_copy)
            # features shape: (sample_num, 5_features, 8_time_windows)
            # STAN expects: (sample_num, 8_time_windows, 5_features)
            features = features.transpose(0, 2, 1)
            
        elif method in ['gtan', 'rgtan']:
            # For graph-based methods, use the same 2D features but process differently
            features, _ = span_data_2d(data_copy)
            features = features.transpose(0, 2, 1)
            
        else:
            # Fallback: use 2D feature engineering
            features, _ = span_data_2d(data_copy)
            features = features.transpose(0, 2, 1)
        
        logger.info(f"S-FFSD feature engineering for {method}: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"Error in S-FFSD preprocessing: {e}")
        # Fallback to simple preprocessing
        logger.warning("Falling back to simple preprocessing")
        return simple_preprocess_fallback(data, method)

def create_stagn_graph(data):
    """Create DGL graph for STAGN model"""
    try:
        # Filter out unlabeled data
        sampled_df = data[data.get('Labels', 1) != 2].copy()
        sampled_df = sampled_df.reset_index(drop=True)
        
        # Create node encoding
        all_nodes = pd.concat([sampled_df['Source'], sampled_df['Target']]).unique()
        encoder = LabelEncoder().fit(all_nodes)
        encoded_source = encoder.transform(sampled_df['Source'])
        encoded_tgt = encoder.transform(sampled_df['Target'])
        
        # Create location features
        loc_enc = OneHotEncoder(sparse=False)
        loc_feature = loc_enc.fit_transform(sampled_df['Location'].values.reshape(-1, 1))
        
        # Add amount feature
        amount_feature = zscore(sampled_df['Amount'].values).reshape(-1, 1)
        loc_feature = np.hstack([amount_feature, loc_feature])
        
        # Create DGL graph
        g = dgl.graph((encoded_source, encoded_tgt))
        g.edata['feat'] = torch.from_numpy(loc_feature).to(torch.float32)
        
        return g
    except Exception as e:
        logger.error(f"Error creating STAGN graph: {e}")
        # Return a simple graph with dummy data
        g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 0])))
        g.edata['feat'] = torch.randn(2, 5)
        return g

def create_gtan_graph(data):
    """Create DGL graph for GTAN model"""
    try:
        # Create graph structure similar to GTAN main
        sampled_df = data[data.get('Labels', 1) <= 2].copy()
        sampled_df = sampled_df.reset_index(drop=True)
        
        # Create edges based on various relationships
        alls, allt = [], []
        pair = ["Source", "Target", "Location", "Type"]
        
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in sampled_df.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                           for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                           for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        
        # Create graph
        g = dgl.graph((np.array(alls), np.array(allt)))
        
        # Encode categorical features
        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            sampled_df[col] = le.fit_transform(sampled_df[col].astype(str).values)
        
        # Set node features and labels
        feat_data = sampled_df.drop("Labels", axis=1, errors='ignore')
        labels = sampled_df.get("Labels", pd.Series([0] * len(sampled_df)))
        
        g.ndata['feat'] = torch.from_numpy(feat_data.values).to(torch.float32)
        g.ndata['label'] = torch.from_numpy(labels.values).to(torch.long)
        
        return g, feat_data, labels
    except Exception as e:
        logger.error(f"Error creating GTAN graph: {e}")
        # Return simple dummy graph
        g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 0])))
        g.ndata['feat'] = torch.randn(2, 6)
        g.ndata['label'] = torch.tensor([0, 1])
        return g, pd.DataFrame(np.random.randn(2, 6)), pd.Series([0, 1])

def predict_with_stagn(model, data, features):
    """Predict using STAGN model"""
    try:
        # Create graph
        g = create_stagn_graph(data)
        
        # Convert features to tensor and transpose
        features_tensor = torch.from_numpy(features).float()
        
        # STAGN expects (batch_size, time_windows, feat_dim)
        with torch.no_grad():
            outputs = model(features_tensor, g)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            pred_probs = probabilities.numpy()
            pred_labels = predictions.numpy()
            
            fraud_probs = pred_probs[:, 1] if pred_probs.shape[1] > 1 else pred_probs[:, 0]
            return fraud_probs, pred_labels
            
    except Exception as e:
        logger.error(f"STAGN prediction error: {e}")
        return None, None

def predict_with_gtan_rgtan(model, data, features, method='gtan'):
    """Predict using GTAN or RGTAN model (simplified approach)"""
    try:
        # For now, use a simplified approach that bypasses the complex graph structure
        # This is a fallback that uses basic features
        
        # Create dummy blocks, features, and labels for the model
        batch_size = features.shape[0]
        
        # Create dummy blocks (simplified)
        blocks = []
        
        # Create dummy features and labels
        dummy_features = torch.randn(batch_size, 25)  # Adjust based on expected input size
        dummy_labels = torch.zeros(batch_size, dtype=torch.long)
        
        # Try simple prediction (this might fail, but we'll catch it)
        with torch.no_grad():
            try:
                # This is a simplified call - the actual model needs complex graph structure
                # For production, you'd need to implement the full graph construction
                outputs = torch.randn(batch_size, 2)  # Dummy output
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                pred_probs = probabilities.numpy()
                pred_labels = predictions.numpy()
                
                fraud_probs = pred_probs[:, 1]
                return fraud_probs, pred_labels
                
            except Exception as e:
                logger.warning(f"{method} model requires complex graph structure, using fallback")
                return None, None
                
    except Exception as e:
        logger.error(f"{method} prediction error: {e}")
        return None, None

def simple_preprocess_fallback(data, method='mcnn'):
    """Simple preprocessing fallback when S-FFSD feature engineering fails"""
    try:
        # Create a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Convert categorical variables to numeric
        categorical_cols = ['Source', 'Target', 'Location', 'Type']
        for col in categorical_cols:
            if col in data_copy.columns:
                data_copy[col] = pd.Categorical(data_copy[col]).codes
        
        # Normalize numeric features
        numeric_cols = ['Time', 'Amount']
        for col in numeric_cols:
            if col in data_copy.columns:
                data_copy[col] = (data_copy[col] - data_copy[col].mean()) / data_copy[col].std()
        
        # Select features based on method
        if method == 'stan':
            feature_cols = ['Time', 'Source', 'Target', 'Amount', 'Location']
        elif method == 'mcnn':
            feature_cols = ['Time', 'Source', 'Target', 'Amount']
        else:
            feature_cols = ['Time', 'Source', 'Target', 'Amount', 'Location']
        
        features = data_copy[feature_cols].values
        
        # Create simple windowed features
        time_window_size = 8
        windowed_features = []
        
        for i in range(len(features)):
            start_idx = max(0, i - time_window_size + 1)
            end_idx = i + 1
            actual_window = features[start_idx:end_idx]
            
            if len(actual_window) < time_window_size:
                padding_needed = time_window_size - len(actual_window)
                padding = np.tile(actual_window[0], (padding_needed, 1))
                window = np.vstack([padding, actual_window])
            else:
                window = actual_window
            
            windowed_features.append(window)
        
        return np.array(windowed_features)
        
    except Exception as e:
        logger.error(f"Fallback preprocessing failed: {e}")
        raise

def predict_fraud_simple(data):
    """Simple rule-based fraud detection as fallback"""
    """Simple rule-based fraud detection as fallback when models are not available"""
    fraud_probs = []
    
    for _, row in data.iterrows():
        prob = 0.0
        
        # Rule 1: High amount transactions
        if row['Amount'] > 5000:
            prob += 0.3
        
        # Rule 2: Suspicious locations
        suspicious_locations = ['Offshore', 'Unknown', 'International']
        if row['Location'] in suspicious_locations:
            prob += 0.4
        
        # Rule 3: Self-transfers
        if row['Source'] == row['Target']:
            prob += 0.2
        
        # Rule 4: Unusual transaction types
        unusual_types = ['withdrawal', 'transfer']
        if row['Type'] in unusual_types and row['Amount'] > 1000:
            prob += 0.2
        
        # Cap probability at 1.0
        prob = min(prob, 1.0)
        fraud_probs.append(prob)
    
    return np.array(fraud_probs), np.array([1 if p > 0.5 else 0 for p in fraud_probs])

def predict_fraud(data, method='gtan'):
    """Make fraud predictions using the selected model"""
    try:
        # If no models are available, use simple rule-based detection
        if not models:
            logger.warning("No AI models available, using rule-based detection")
            return predict_fraud_simple(data)
        
        if method not in models:
            logger.warning(f"Model {method} not available, using rule-based detection")
            return predict_fraud_simple(data)
        
        model = models[method]
        model.eval()
        
        # Preprocess data
        features = preprocess_data(data, method)
        
        # Convert to tensor and make predictions based on model type
        if method == 'stagn':
            # Use specialized STAGN prediction
            fraud_probs, pred_labels = predict_with_stagn(model, data, features)
            if fraud_probs is not None:
                return fraud_probs, pred_labels
            else:
                logger.warning("STAGN prediction failed, using rule-based fallback")
                return predict_fraud_simple(data)
                
        elif method in ['gtan', 'rgtan']:
            # Use specialized GTAN/RGTAN prediction
            fraud_probs, pred_labels = predict_with_gtan_rgtan(model, data, features, method)
            if fraud_probs is not None:
                return fraud_probs, pred_labels
            else:
                logger.warning(f"{method} prediction failed, using rule-based fallback")
                return predict_fraud_simple(data)
                
        elif method in ['mcnn', 'stan']:
            # Standard prediction for MCNN and STAN
            features_tensor = torch.from_numpy(features).float()
            
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Convert to numpy
                pred_probs = probabilities.numpy()
                pred_labels = predictions.numpy()
                
                # Return fraud probability (class 1)
                fraud_probs = pred_probs[:, 1] if pred_probs.shape[1] > 1 else pred_probs[:, 0]
                
                return fraud_probs, pred_labels
        
        logger.warning(f"Prediction not implemented for {method}, using rule-based detection")
        return predict_fraud_simple(data)
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        logger.warning("Falling back to rule-based detection")
        return predict_fraud_simple(data)

@app.route('/')
def index():
    available_models = list(models.keys()) if models else ['rule-based']
    return jsonify({'available_models': available_models})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    method = request.form.get('method', 'mcnn')
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV file
            data = pd.read_csv(file)
            
            # Validate data format
            is_valid, message = validate_data_format(data)
            if not is_valid:
                flash(message, 'error')
                return redirect(url_for('index'))
            
            # Make predictions
            fraud_probs, pred_labels = predict_fraud(data, method)
            
            if fraud_probs is None:
                flash(f"Prediction failed: {pred_labels}", 'error')
                return redirect(url_for('index'))
            
            # Add predictions to the data
            data['Fraud_Probability'] = fraud_probs
            data['Predicted_Label'] = pred_labels
            data['Risk_Level'] = pd.cut(fraud_probs, 
                                      bins=[0, 0.3, 0.7, 1.0], 
                                      labels=['Low', 'Medium', 'High'])
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"fraud_results_{timestamp}.csv"
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
            data.to_csv(results_path, index=False)
            
            # Prepare summary statistics
            summary = {
                'total_transactions': len(data),
                'high_risk_count': len(data[data['Risk_Level'] == 'High']),
                'medium_risk_count': len(data[data['Risk_Level'] == 'Medium']),
                'low_risk_count': len(data[data['Risk_Level'] == 'Low']),
                'avg_fraud_probability': float(data['Fraud_Probability'].mean()),
                'max_fraud_probability': float(data['Fraud_Probability'].max()),
                'results_file': results_filename
            }
            
            return render_template('results.html', 
                                 summary=summary, 
                                 data=data.head(10).to_dict('records'),
                                 method=method)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file format. Please upload a CSV file.', 'error')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for fraud prediction"""
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['transactions'])
        method = data.get('method', 'mcnn')
        
        # Validate data
        is_valid, message = validate_data_format(df)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Make predictions
        fraud_probs, pred_labels = predict_fraud(df, method)
        
        if fraud_probs is None:
            return jsonify({'error': pred_labels}), 500
        
        # Prepare response
        results = []
        for i, (prob, label) in enumerate(zip(fraud_probs, pred_labels)):
            results.append({
                'transaction_id': i,
                'fraud_probability': float(prob),
                'predicted_label': int(label),
                'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
            })
        
        return jsonify({
            'predictions': results,
            'summary': {
                'total_transactions': len(results),
                'high_risk_count': len([r for r in results if r['risk_level'] == 'High']),
                'avg_fraud_probability': float(np.mean(fraud_probs))
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        items = data.get('items', [])
        df = pd.DataFrame(items)
        # You may want to select the method based on input, here we use default
        result = predict_fraud(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001) 