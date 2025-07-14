
# 🔐 NeuroPay Backend & Fraud Detection Engine

This is the backend and machine learning system powering **NeuroPay**, a secure transaction platform that verifies users with both behavioral biometrics and financial fraud detection.

---

## 📁 Project Structure

```
.
├── Gesture_models/                  # Trained gesture-based biometric models
│   ├── CompleteModel.py
│   ├── SiameseNet.py
│   ├── decision_model.pth
│   ├── latest_model_epoch_46.pth
│   ├── DecisionNetwork.py
│   └── __init__.py
│
├── qdrant/                          # Qdrant integration for vector DB
│   └── model.py
│
├── src/
│   ├── controller/                  # Business logic controllers
│   │   ├── fraud.controller.js
│   │   ├── payment.controller.js
│   │   ├── product.controller.js
│   │   └── user.controller.js
│   │
│   ├── db/                          # DB connection
│   │   └── connectToMongoDB.js
│   │
│   ├── middleware/                 # Auth and verification middlewares
│   │   └── authMiddleware.js
│   │
│   ├── models/                      # Mongoose data models
│   │   ├── payment.model.js
│   │   ├── product.model.js
│   │   └── user.model.js
│   │
│   ├── routes/                      # Express routes
│   │   ├── fraud.routes.js
│   │   ├── payment.routes.js
│   │   ├── productRoute.js
│   │   └── userRoute.js
│   │
│   └── utils/
│       └── generateToken.js        # JWT token helper
│
├── antifraud/                      # Python-based fraud detection engine
│   ├── config/
│   ├── feature_engineering/
│   ├── methods/
│   ├── models/
│   ├── main.py                     # Flask app entry for fraud service
│   └── requirements.txt
│
├── biometric_vectors.csv           # Stored gesture embeddings
├── fraud_detection_service.py      # Legacy or wrapper entry point
├── gesture_app.py                  # Biometric prediction service
├── package.json
├── README.md
├── requirements.txt
├── server.js                       # Node.js/Express API entry point
└── .gitignore
```

---

## 🚀 Features

- 🔐 **Two-Factor Verification**: Combines gesture biometrics and transaction behavior analytics.
- 🧠 **Few-Shot Learned Siamese Model**: Lightweight user identification based on motion vectors.
- 🗂 **Modular Architecture**: Separate layers for model inference, API routes, controllers, and fraud engine.
- 🧪 **Vector DB (Qdrant)**: For efficient gesture vector storage and similarity search.
- 💳 **Secure Payments**: Stripe and UPI integrated, with fraud screening before processing.

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroPay.git
cd NeuroPay
```

### 2. Install Node.js Backend Dependencies

```bash
cd backend
npm install
```

### 3. Install Python Fraud Detection Dependencies

```bash
cd antifraud
pip install -r requirements.txt
cd ..
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the backend root:

```env
MONGO_DB_URI=your_mongodb_uri
PORT=5000

JWT_SECRET=your_jwt_secret
NODE_ENV=development

STRIPE_SECRET_KEY=your_stripe_key
FRAUD_DETECTION_SERVICE_URL=http://localhost
FRAUD_DETECTION_SERVICE_PORT=8000
```

---

## 🧠 Running the System

### Start Node.js Server

```bash
node server.js
npm run dev
```
---
### Start Python Fraud Detection Service 

```bash
python app.py
python fraud_detection_service.py
```

---

### Start Qdrant Service - Docker

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---


### Start Biometrics Service  

```bash

python gesture_app.py
```
---


## 📬 Contact

For queries or contributions, contact [arpang@iitbhilai.ac.in, chetan@iitbhilai.ac.in, shivam@iitbhilai.ac.in].

---
