# Noto
Spatial Computer Vision & Multi-Modal AI Study Workspace

Noto bridges the gap between raw physical notes and structured digital intelligence. By applying geometric matrix transforms to photos of whiteboards, textbooks, and handwritten notes, Noto flattens, extracts, and vectors spatial context into an interactive AI workspace.

## 🌟Core Architecture
- **Perspective & Geometry Engine:** Custom OpenCV pipeline for contour detection and 4-point perspective warp transformations.
- **Layout-Aware Extraction:** Multi-modal OCR for spatial bounding-box segmentation (Text, Formulas, Diagrams).
- **Spatial RAG Engine:** Vector embeddings paired with spatial coordinates for hyper-targeted context retrieval.
- **Interactive Workspace:** Split-screen Next.js canvas with interactive image highlights and structured AI agent workflows.

## 🛠️Tech Stack
- **Frontend:** Next.js (React), Tailwind CSS, HTML5 Canvas / SVG
- **Backend:** FastAPI (Python), OpenCV, Surya OCR / PaddleOCR
- **Vector Store & DB:** PostgreSQL (`pgvector`) / ChromaDB
- **Orchestration:** LangChain / LlamaIndex, Pydantic
