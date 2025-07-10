# Walmart Hack Backend - Setup Instructions

This project consists of two main services:
1. **Node.js Backend** (Port 5002) - Main API server
2. **Python Fraud Detection Service** (Port 5001) - ML-based fraud detection

## Prerequisites

- Node.js (v16 or higher)
- Python 3.11
- MongoDB connection

## Setup Instructions

### 1. Node.js Backend Setup

```bash
# Install Node.js dependencies
npm install

# Start the Node.js server
npm i
npm run dev
```

The Node.js backend will run on `http://localhost:5002`

### 2. Python Virtual Environment Setup

#### Creating Virtual Environment (First time only)

```bash
# Create virtual environment
python3 -m venv fraud_venv

# Activate the virtual environment
source fraud_venv/bin/activate

# Install Python dependencies
pip install -r fraud_requirements.txt
```

#### Activating Existing Virtual Environment

```bash
# Activate the virtual environment
source fraud_venv/bin/activate

# Verify activation (you should see (fraud_venv) in your terminal prompt)
which python
# Should show: /path/to/your/project/fraud_venv/bin/python
```

#### Deactivating Virtual Environment

```bash
# When you're done working
deactivate
```

### 3. Start the Fraud Detection Service

```bash
# Make sure virtual environment is activated
source fraud_venv/bin/activate

# Start the fraud detection service
fraud_venv/bin/python fraud_detection_service.py
```

The fraud detection service will run on `http://localhost:5001`

## Running Both Services

**Important:** You need to run both services simultaneously for the application to work properly.

### Option 1: Using Two Terminals

**Terminal 1** - Start the fraud detection service:
```bash
cd /path/to/Walmart_hack_back
source fraud_venv/bin/activate
fraud_venv/bin/python fraud_detection_service.py
```

**Terminal 2** - Start the Node.js backend:
```bash
cd /path/to/Walmart_hack_back
npm run dev
```

### Option 2: Using Background Process

```bash
# Start fraud detection service in background
source fraud_venv/bin/activate
nohup fraud_venv/bin/python fraud_detection_service.py &

# Start Node.js backend
npm run dev
```

## Service Architecture

```
Frontend/Client
       ↓
Node.js Backend (Port 5002)
       ↓ HTTP requests
Python Fraud Detection Service (Port 5001)
       ↓
ML Models (RGTAN, MCNN, STAN, etc.)
```

## API Endpoints

### Node.js Backend (Port 5002)
- User authentication
- Payment processing
- Product management
- Order management

### Python Fraud Detection Service (Port 5001)
- `POST /predict` - Single transaction fraud detection
- `POST /predict/batch` - Batch fraud detection
- `GET /health` - Service health check

## Environment Variables

Make sure you have a `.env` file with the required environment variables:
- `MONGO_DB_URI`
- `PORT`
- `STRIPE_SECRET_KEY`
- `JWT_SECRET`

## Troubleshooting

### Common Issues

1. **Virtual Environment Issues**
   ```bash
   # Check if virtual environment is activated
   echo $VIRTUAL_ENV
   # Should show: /path/to/your/project/fraud_venv
   
   # If not activated
   source fraud_venv/bin/activate
   ```

2. **Port already in use**
   ```bash
   # Kill existing processes using the ports
   lsof -ti:5001 | xargs kill -9  # Kill processes on port 5001
   lsof -ti:5002 | xargs kill -9  # Kill processes on port 5002
   ```

3. **Python module not found**
   ```bash
   # Make sure virtual environment is activated
   source fraud_venv/bin/activate
   
   # Reinstall dependencies if needed
   pip install -r fraud_requirements.txt
   ```

4. **Models not loading**
   ```bash
   # Check if model files exist
   ls -la antifraud/models/
   
   # Should show .pth files for different models
   ```

## Virtual Environment Benefits

- **Isolation**: Keeps project dependencies separate from system Python
- **Version Control**: Ensures consistent package versions across environments
- **No Conflicts**: Prevents package conflicts between different projects
- **Easy Cleanup**: Can delete entire `fraud_venv` folder to remove all dependencies

## Notes

- The fraud detection service loads 5 ML models on startup (MCNN, STAN, GTAN, STAGN, RGTAN)
- The Node.js backend communicates with the Python service via HTTP requests
- All payment transactions are automatically checked for fraud using the RGTAN model
- Virtual environment folder (`fraud_venv/`) is ignored by Git (in `.gitignore`)
