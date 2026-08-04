import uuid
import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# load environment variables from .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# safety check to prevent falling back to localhost
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set in environment variables / .env file.")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "noto_pages"

# FastEmbed automatically downloads and caches the lightweight ONNX model locally
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def init_vector_db():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def index_ocr_blocks(text_blocks: list, page_id: str) -> int:
    """Embeds OCR text blocks and upserts them as points into Qdrant."""
    init_vector_db()  # to ensure 'noto_pages' collection exists before indexing

    if not text_blocks:
        return 0

    texts = [b["text"] for b in text_blocks]
    # Generate embeddings generator and convert to list
    embeddings = list(embedding_model.embed(texts))

    points = []
    for block, vector in zip(text_blocks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "page_id": page_id,
                    "text": block.get("text"),
                    "bbox": block.get("bbox"),
                    "label": block.get("label"),
                    "confidence": block.get("confidence")
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)