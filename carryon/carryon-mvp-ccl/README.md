# CarryOn MVP

A memory-augmented AI assistant with vector embeddings, FAISS retrieval, and tone alignment capabilities.

## Features

- **Memory Management**: Store and retrieve contextual memories
- **Vector Search**: FAISS-based semantic search with SentenceTransformers
- **TF-IDF Fallback**: Offline operation when embeddings unavailable
- **Tone Alignment**: Evaluate reply tone against persona voice
- **Graph-based Ranking**: Multi-factor memory ranking with graph centrality
- **Electron UI**: Desktop application for easy interaction

## Quick Start

### Server Setup

```bash
cd server
pip install -e .
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Desktop App

```bash
cd apps/desktop-electron
npm install
npm start
```

## API Endpoints

### Memories
- `POST /v1/memories` - Add new memory
- `GET /v1/memories` - List all memories

### Soulpacks
- `POST /v1/soulpacks` - Load persona configuration
- `GET /v1/soulpacks` - List loaded soulpacks

### Prime & Query
- `POST /v1/prime` - Prime context with query
- `POST /v1/prime/query` - Query with primed context

### Embeddings & Vector Search
- `POST /v1/embeddings/rebuild` - Rebuild vector index from memories
- `GET /v1/embeddings/stats` - Get index statistics

### Tone Alignment
- `POST /v1/tone/evaluate` - Evaluate tone alignment for proposed reply

## Embeddings & Tone

### Rebuild Vector Index
```bash
curl -X POST http://localhost:8000/v1/embeddings/rebuild
curl http://localhost:8000/v1/embeddings/stats
```

### Evaluate Tone Alignment
```bash
curl -X POST http://localhost:8000/v1/tone/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there!"}'
```

## Code Analysis (CCL)

### Run CCL Analysis on Specific Path
```bash
curl -X POST http://localhost:8000/v1/tools/ccl-analyze \
  -H 'Content-Type: application/json' \
  -d '{"path":"server/app/retrieval/vector_index.py","samples":100,"seed":42}'
```

### Quick CCL Scan of Core Files
```bash
curl -X POST http://localhost:8000/v1/tools/ccl-quick
```

### CCL Features
- **Idempotence Check**: Tests if f(f(x)) == f(x)
- **Commutativity Check**: Tests if f(x,y) == f(y,x) for binary operations
- **Associativity Check**: Tests if f(f(x,y),z) == f(x,f(y,z))
- **Pipeline Coherence**: Tests f∘g vs g∘f composition
- **Entropy Analysis**: Identifies high-entropy, low-coherence code zones
- **Ghost Detection**: Ranks functions by "ghost likelihood" based on coherence metrics

## Architecture

### Vector Index
- **Primary**: FAISS with SentenceTransformers embeddings
- **Fallback**: TF-IDF vectorization when offline
- **Storage**: FAISS index files + numpy arrays for fallback

### Memory Ranking
1. **Vector Preselection**: FAISS search for semantic similarity
2. **Multi-factor Scoring**: 
   - Semantic similarity (α)
   - Recency (β) 
   - Graph centrality (γ)
   - Entropy adjustment (δ)

### Tone Alignment
- **Vector Similarity**: Compare reply to persona voice descriptor
- **Style Metrics**: Adverb ratio, sentence length, exclamation rate
- **Rule-based Suggestions**: Contextual improvement tips

## Configuration

The system automatically detects available dependencies:
- `sentence-transformers` + `faiss` → Full vector search
- `scikit-learn` only → TF-IDF fallback
- Neither → Hash-based fallback

## Development

### Dependencies
- FastAPI + SQLModel for API
- SentenceTransformers for embeddings
- FAISS for vector search
- scikit-learn for TF-IDF fallback
- numpy for numerical operations

### Project Structure
```
carryon-mvp/
├── server/
│   └── app/
│       ├── retrieval/     # Vector search & ranking
│       ├── tone/          # Tone alignment
│       ├── routers/       # API endpoints
│       ├── models/        # Data models
│       └── config/        # Configuration
└── apps/
    └── desktop-electron/  # Electron UI
```

## License

Apache 2.0 