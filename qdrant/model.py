import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from Gesture_models.CompleteModel import CompleteModel
import csv
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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
            vectors_config=VectorParams(size=9216, distance=Distance.COSINE)
        )

# Optional global flag
models_loaded = False

def initialize_models():
    global models_loaded
    try:
        # load_models()  # Uncomment if needed
        initialize_qdrant()
        models_loaded = True
        print("✅ Qdrant initialized")
        return True
    except Exception as e:
        print(f"❌ Init error: {e}")
        return False



def store_vector_in_qdrant(user_id, vector, source="biometric-app", max_vectors_per_user=50):
    """
    Store vector with is_fraud determined via model after 15 entries.
    """
    timestamp = datetime.datetime.now().isoformat()

    try:
        # Step 1: Get all previous vectors for user
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


        csv_file = "biometric_vectors.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_id", "timestamp" ] + [f"v{i}" for i in range(len(vector))])
            writer.writerow([user_id, timestamp] + vector)

        # Step 2: Determine is_fraud
        if vector_count < 15:
            is_fraud = False
        else:
            # Get one reference vector with is_fraud == False
            reference = next((p.vector for p in existing_points if p.payload.get("is_fraud") is False), None)
            if reference is None:
                print("⚠️ No reference vector with is_fraud=False found. Defaulting to is_fraud=True.")
                is_fraud = True
            else:
                # Use model to compare vectors
                model = CompleteModel()
                is_fraud = model.solve(reference, vector)


        # Step 3: Delete 5 oldest if over max
        if vector_count >= max_vectors_per_user:
            existing_points.sort(key=lambda p: p.payload.get("timestamp", ""))
            to_delete = [p.id for p in existing_points[:5]]
            qdrant.delete(collection_name=COLLECTION_NAME, points_selector={"points": to_delete})
            print(f"🧹 Deleted {len(to_delete)} old vectors for user {user_id}")

        # Step 4: Store the new vector
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

        print(f"✅ Stored vector for {user_id} | is_fraud: {is_fraud}")
        return True, None

    except Exception as e:
        print(f"❌ Error storing vector: {e}")
        return False, str(e)
    


def print_all_vectors(collection_name=COLLECTION_NAME, limit=30):
    """
    Fetch and print all points (vectors + metadata) from a Qdrant collection.
    """
    try:
        response = qdrant.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )

        points = response[0]  # scroll returns a tuple: (points, next_page_offset)
        print(f"📦 Found {len(points)} vectors in collection '{collection_name}':\n")

        for i, point in enumerate(points):
            print(f"--- Vector {i+1} ---")
            print(f"ID: {point.id}")
            print(f"Vector: {point.vector[:5]}... ({len(point.vector)} dims)")  # Print only first 5 dims
            print(f"Payload: {point.payload}")
            print("")

    except Exception as e:
        print(f"❌ Error fetching vectors: {e}")

def delete_all_vectors_for_user(user_id: str):
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
        )
        print(f"🧹 Successfully deleted all vectors for user_id: {user_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete vectors for user_id {user_id}: {e}")
        return False


def get_fraud_summary_for_user(user_id: str):
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
        with_vectors=True,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    )

    if not points:
        return {
            "message": f"No vectors found for user {user_id}",
            "fraud_true": 0,
            "fraud_false": 0,
            "latest_vector": None
        }

    fraud_true = sum(1 for p in points if p.payload.get("is_fraud") is True)
    fraud_false = sum(1 for p in points if p.payload.get("is_fraud") is False)
    latest = max(points, key=lambda p: p.payload.get("timestamp", ""))

    return {
        "user_id": user_id,
        "fraud_true": fraud_true,
        "fraud_false": fraud_false
    }


print_all_vectors()
# delete_all_vectors_for_user("6866ba016004cc970eb8b429")