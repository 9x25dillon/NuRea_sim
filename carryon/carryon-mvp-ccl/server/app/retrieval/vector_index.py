import os, json
from typing import List, Tuple, Optional
import numpy as np

# Runtime-optional imports
_st_model = None
_tfidf = None
_use_tfidf = False

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

from ..config import settings
from ..models.memory_event import MemoryEvent

IDX_PATH = settings.faiss_index_path
IDS_PATH = IDX_PATH + ".ids.json"
MODEL_NAME = settings.embeddings_model

def _load_model():
    global _st_model, _use_tfidf
    if _st_model is not None or _use_tfidf:
        return
    if SentenceTransformer is None:
        _use_tfidf = True
        return
    try:
        _st_model = SentenceTransformer(MODEL_NAME)
    except Exception:
        # Model download failed (offline?) – fallback to TF-IDF
        _use_tfidf = True

def _embed_texts(texts: List[str]) -> np.ndarray:
    _load_model()
    if _use_tfidf:
        return _tfidf_embed(texts)
    return np.asarray(_st_model.encode(texts, normalize_embeddings=True), dtype="float32")

# --- TF-IDF fallback (no internet) ---
def _tfidf_embed(texts: List[str]) -> np.ndarray:
    global _tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    if _tfidf is None:
        _tfidf = TfidfVectorizer(max_features=4096)
        _tfidf.fit(texts)
    X = _tfidf.transform(texts).astype("float32").toarray()
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return (X / norms).astype("float32")

# --- FAISS helpers ---
def _save_ids(ids: List[str]):
    os.makedirs(os.path.dirname(IDS_PATH), exist_ok=True)
    with open(IDS_PATH, "w") as f:
        json.dump(ids, f)

def _load_ids() -> List[str]:
    if not os.path.exists(IDS_PATH):
        return []
    return json.load(open(IDS_PATH))

def _save_index(index, dim: int):
    os.makedirs(os.path.dirname(IDX_PATH), exist_ok=True)
    if faiss is None:
        # Persist as npy for TF-IDF fallback or no-faiss env
        np.save(IDX_PATH + ".npy", index)  # type: ignore
    else:
        faiss.write_index(index, IDX_PATH)

def _load_index(dim: int):
    if faiss is None:
        npy = IDX_PATH + ".npy"
        if os.path.exists(npy):
            return np.load(npy)
        return None
    if os.path.exists(IDX_PATH):
        return faiss.read_index(IDX_PATH)
    return None

def rebuild_index(mems: List[MemoryEvent]) -> dict:
    texts = [m.data for m in mems]
    ids = [m.event_id for m in mems]
    if not texts:
        return {"ok": True, "count": 0}

    X = _embed_texts(texts)
    dim = X.shape[1]

    if faiss is None or _use_tfidf:
        # Save dense matrix as fallback store
        _save_index(X, dim)
    else:
        index = faiss.IndexFlatIP(dim)  # cosine with normalized embeddings
        index.add(X)
        _save_index(index, dim)

    _save_ids(ids)
    return {"ok": True, "count": len(ids), "dim": dim, "faiss": faiss is not None and not _use_tfidf}

def stats() -> dict:
    ids = _load_ids()
    has_faiss = faiss is not None and os.path.exists(IDX_PATH)
    has_npy = os.path.exists(IDX_PATH + ".npy")
    return {"count": len(ids), "faiss": has_faiss, "fallback": has_npy}

def search(query: str, top_n: int = 24) -> List[Tuple[str, float]]:
    ids = _load_ids()
    if not ids:
        return []
    # Load store
    _load_model()
    if faiss is None or _use_tfidf:
        X = np.load(IDX_PATH + ".npy")
        q = _embed_texts([query])[0:1]
        sims = (X @ q.T).ravel()  # cosine similarity on normalized vectors
        idx = np.argsort(-sims)[:top_n]
        return [(ids[i], float(sims[i])) for i in idx]

    index = _load_index(dim=None)  # faiss reads dim internally
    if index is None:
        return []
    q = _embed_texts([query])
    sims, nn = index.search(q, top_n)
    sims = sims.ravel().tolist()
    nn = nn.ravel().tolist()
    out = []
    for i, s in zip(nn, sims):
        if i == -1 or i >= len(ids):
            continue
        out.append((ids[i], float(s)))
    return out 