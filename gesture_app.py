import os
import datetime
import jwt
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv  # NEW
from qdrant.model import store_vector_in_qdrant, initialize_qdrant,initialize_models  # Adjust import based on your structure
load_dotenv() 


app = Flask(__name__)
CORS(app)

JWT_SECRET = os.environ.get("JWT_SECRET", "your_secret_key")  # fallback for dev


# In-memory store: {user_id: [vectors]}
vector_store = {}

# JWT middleware
def is_authenticated(f):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ")[1]

        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.user_id = decoded.get("userID")
            if not g.user_id:
                return jsonify({"error": "Invalid token"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__  # Fix Flask route bug
    return wrapper


@app.route("/api/biometrics", methods=["POST"])
@is_authenticated
def receive_vector():
    data = request.get_json()
    vector = data.get("vector")

    if not vector or not isinstance(vector, list):
        return jsonify({"error": "Invalid or missing vector"}), 400

    user_id = g.user_id
    timestamp = datetime.datetime.now().isoformat()
    vector_len = len(vector)

    print(f"📥 Received vector for user {user_id} at {timestamp} (length: {vector_len})")

    # Directly store in Qdrant
    success, error = store_vector_in_qdrant(user_id, vector)

    if not success:
        print(f"⚠️ Qdrant insert failed: {error}")
        return jsonify({"message": "Qdrant insert failed", "error": error}), 500

    return jsonify({"message": "Vector stored in Qdrant", "user": user_id}), 200



if __name__ == "__main__":
    initialize_qdrant()
    initialize_models()
    print("✅ Qdrant and models initialized")
    app.run(debug=True, host="0.0.0.0", port=5001)