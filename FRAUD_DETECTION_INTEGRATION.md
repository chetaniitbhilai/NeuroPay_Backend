# Fraud Detection Integration for Walmart Hack Backend

This document outlines the integration of the RGTAN fraud detection model into the Walmart Hack Backend.

## Overview

The integration includes:
- **Python Fraud Detection Service**: Runs the RGTAN model for fraud detection
- **Enhanced Payment Schema**: Stores fraud detection results
- **Fraud Detection API**: Provides fraud analytics and manual checking
- **Automatic Fraud Checking**: All payments are automatically checked for fraud

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Node.js       │    │   Python        │
│   (React)       │───▶│   Backend       │───▶│   Fraud Service │
│                 │    │   (Port 5000)   │    │   (Port 5001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   MongoDB       │
                       │   (Payments +   │
                       │   Fraud Data)   │
                       └─────────────────┘
```

## Setup Instructions

### 1. Prerequisites
- Node.js 18+
- Python 3.7+
- MongoDB
- ✅ All antifraud model files are already copied to this directory

### 2. Installation
```bash
# All required files are already copied to the current directory:
# - antifraud/ (fraud detection models and code)
# - app.py (main fraud detection logic)
# - fraud_venv/ (Python virtual environment with dependencies)
# - fraud_requirements.txt (Python dependencies)

# Run the setup script
./setup_fraud_detection.sh
```

### 3. Environment Variables
Add to your `.env` file:
```
FRAUD_DETECTION_SERVICE_URL=http://localhost:5001
```

### 4. Start Services
```bash
# Start both services
./start_services.sh

# Or start individually:
# Terminal 1: Python service (using the copied virtual environment)
source fraud_venv/bin/activate
python3 fraud_detection_service.py

# Terminal 2: Node.js backend
npm run dev
```

## API Endpoints

### Payment Endpoints (Enhanced)
All existing payment endpoints now include fraud detection:

- `POST /api/payments/create-intent` - Returns fraud results
- `POST /api/payments/upi` - Returns fraud results  
- `POST /api/payments/mark-success` - Existing functionality
- `GET /api/payments/history` - Now includes fraud data

### New Fraud Detection Endpoints

#### Get Fraud Analytics
```bash
GET /api/fraud/analytics
Authorization: Bearer <token>
```

Response:
```json
{
  "overview": {
    "totalPayments": 1000,
    "fraudPayments": 45,
    "checkedPayments": 1000,
    "fraudRate": "4.50"
  },
  "riskDistribution": [
    { "_id": "LOW", "count": 800 },
    { "_id": "MEDIUM", "count": 155 },
    { "_id": "HIGH", "count": 45 }
  ],
  "fraudTrends": [...],
  "fraudByMethod": [...],
  "highRiskTransactions": [...]
}
```

#### Check Single Transaction
```bash
POST /api/fraud/check
Content-Type: application/json
Authorization: Bearer <token>

{
  "paymentId": "payment_id_here"
}
```

Response:
```json
{
  "message": "Fraud check completed",
  "fraudResult": {
    "status": "success",
    "fraud_probability": 0.85,
    "prediction": 1,
    "risk_level": "HIGH",
    "is_fraud": true,
    "model_used": "rgtan"
  },
  "payment": { ... }
}
```

## Payment Schema Changes

The payment schema now includes fraud detection results:

```javascript
{
  // ... existing fields ...
  
  fraudDetection: {
    isChecked: Boolean,        // Whether fraud check was performed
    fraudProbability: Number,  // Fraud probability (0-1)
    prediction: Number,        // 0 = legitimate, 1 = fraud
    riskLevel: String,         // 'LOW', 'MEDIUM', 'HIGH'
    isFraud: Boolean,          // True if fraud detected
    modelUsed: String,         // Model used ('rgtan')
    checkedAt: Date           // When check was performed
  }
}
```

## Frontend Integration

### Display Fraud Results
```javascript
// After payment creation
const response = await fetch('/api/payments/create-intent', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(paymentData)
});

const { fraudResult } = await response.json();

if (fraudResult?.risk_level === 'HIGH') {
  // Show warning to user
  alert('High fraud risk detected. Please verify transaction.');
}
```

### Admin Dashboard
```javascript
// Get fraud analytics
const analytics = await fetch('/api/fraud/analytics');
const data = await analytics.json();

// Display fraud rate, trends, and high-risk transactions
```

## Fraud Detection Model

The integration uses the **RGTAN (Risk-aware Graph Transformer Attention Network)** model, which:

- Analyzes transaction patterns using graph neural networks
- Considers temporal and spatial features
- Provides probability scores and risk levels
- Achieves high accuracy on financial fraud detection

### Model Performance
- **AUC**: 0.8461
- **F1 Score**: 0.7513
- **Average Precision**: 0.6939

## Monitoring & Logging

- Python service logs: `logs/fraud_service.log`
- Node.js logs: Console output
- MongoDB: Fraud detection results stored in payment documents

## Troubleshooting

### Common Issues

1. **Python service not starting**
   - Activate virtual environment: `source fraud_venv/bin/activate`
   - Check Python dependencies: `pip install -r fraud_requirements.txt`
   - Verify antifraud model files are in the current directory

2. **Model loading errors**
   - Ensure DGL version compatibility (included in fraud_venv)
   - Check PyTorch version matches model requirements
   - Verify all antifraud files are copied correctly

3. **Connection errors**
   - Verify Python service is running on port 5001
   - Check firewall settings
   - Ensure fraud_venv is activated

### Health Check
```bash
# Check if fraud detection service is running
curl http://localhost:5001/health
```

## Security Considerations

- Fraud detection service runs on localhost only
- All endpoints require authentication
- Sensitive fraud data is stored securely in MongoDB
- Model predictions are logged for audit purposes

## Future Enhancements

- Real-time fraud alerts
- Machine learning model retraining
- Integration with external fraud databases
- Advanced risk scoring algorithms
- Fraud pattern visualization

## Support

For issues or questions:
1. Check the logs in `logs/fraud_service.log`
2. Verify all dependencies are installed in `fraud_venv/`
3. Ensure the antifraud model files are in the current directory
4. Check MongoDB connection and schema updates
5. Verify file structure:
   ```
   Walmart_hack_back/
   ├── antifraud/              # Fraud detection models and code
   ├── app.py                  # Main fraud detection logic
   ├── fraud_venv/             # Python virtual environment
   ├── fraud_requirements.txt  # Python dependencies
   ├── fraud_detection_service.py  # Fraud detection service
   └── src/                    # Your Node.js backend
   ```
