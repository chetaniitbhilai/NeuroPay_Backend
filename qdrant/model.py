import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from Gesture_models.CompleteModel import CompleteModel
import csv
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import PointIdsList
import random
import os

# Setup Qdrant
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "biometric-vectors_9216"
model = CompleteModel()

# Ensure collection exists
def initialize_qdrant():
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=9216,distance=Distance.COSINE)
        )

# Optional global flag
models_loaded = False

def initialize_models():
    global models_loaded
    try:
        initialize_qdrant()
        models_loaded = True
        print("Qdrant initialized")
        return True
    except Exception as e:
        print(f"Init error: {e}")
        return False



def store_vector_in_qdrant(user_id, vector, source="biometric-app", max_vectors_per_user=50):
    """
    Store vector with is_fraud determined via model after 15 entries.
    """
    timestamp = datetime.datetime.now().isoformat()

    try:
    
        assert(len(vector) == 9216)
        existing_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=max_vectors_per_user + 20,
            with_payload=True,
            with_vectors=True,
            scroll_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}}
                ]
            }
        )
        vector_count = len(existing_points)

        print(f"Found {vector_count} existing vectors for user {user_id}")
       
        if vector_count < 15:
            is_fraud = False
        else:
            reference = random.choice(existing_points).vector if existing_points else None
            print(f"Reference vector found: {reference is not None}")
            if reference is None:
                print("No reference vector with is_fraud=False found. Defaulting to is_fraud=True.")
                is_fraud = True
            else:
                model = CompleteModel()
                is_fraud = model.solve(reference, vector)
                print(f"is_fraud: {is_fraud}")

        print(f" User {user_id} has {vector_count} existing vectors. is_fraud: {is_fraud}")
        

        if (not is_fraud):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=int(datetime.datetime.now().timestamp() * 1000),
                        vector=vector,
                        payload={
                            "user_id": user_id,
                            "timestamp": timestamp,
                            "is_fraud": is_fraud,
                            "source": source
                        }
                    )
                ]
            )

            print(f" Stored vector for {user_id} | is_fraud: {is_fraud}")
            
            if vector_count >= max_vectors_per_user:
                existing_points.sort(key=lambda p: p.payload.get("timestamp", ""))
                to_delete = [p.id for p in existing_points[:5]]
                print(f" Deleting {len(to_delete)} oldest vectors for user {user_id}")
                qdrant.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=to_delete)
                )
                print(f"Deleted {len(to_delete)} old vectors for user {user_id}")

        return is_fraud,None

    except Exception as e:
        print(f" Error storing vector: {e}")
        return None, str(e)

