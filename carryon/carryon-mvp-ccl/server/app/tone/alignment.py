from typing import Dict, List
import re
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

def _embed(texts: List[str]):
    if SentenceTransformer is None:
        # Simple hashing-based embedding fallback
        return np.stack([_hash_embed(t) for t in texts])
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(texts, normalize_embeddings=True)
    except Exception:
        return np.stack([_hash_embed(t) for t in texts])

def _hash_embed(t: str, dim: int = 256):
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    v = rng.normal(size=(dim,)).astype('float32')
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def style_metrics(text: str) -> Dict[str, float]:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return {"adverb_ratio":0, "avg_sentence_len":0, "exclamation_rate":0}
    adverbs = sum(1 for w in words if w.endswith('ly'))
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    avg_len = (sum(len(s.split()) for s in sentences) / max(1, len(sentences))) if sentences else len(words)
    exclam = text.count('!') / max(1, len(text))
    return {"adverb_ratio": adverbs/max(1,len(words)), "avg_sentence_len": avg_len, "exclamation_rate": exclam}

def evaluate_tone(persona_voice: Dict, candidate_text: str) -> Dict:
    # Vector similarity between candidate and persona style descriptor
    tone_list = persona_voice.get('tone', [])
    rules_list = persona_voice.get('style_rules', [])
    persona_desc = "; ".join(["Tone:"+", ".join(tone_list), "Rules:"+" | ".join(rules_list)])
    P, C = _embed([persona_desc, candidate_text])
    sim = (1.0 + _cos(P, C)) / 2.0  # 0..1

    # Heuristic checks aligned with rules
    metrics = style_metrics(candidate_text)
    tips: List[str] = []
    if any('avoid purple prose' in r.lower() for r in rules_list) and metrics['adverb_ratio'] > 0.12:
        tips.append("Trim adverbs/adjectives to reduce purple prose.")
    if any('use concrete examples' in r.lower() for r in rules_list) and metrics['avg_sentence_len'] > 24:
        tips.append("Shorten sentences and add specific examples.")
    if any('match user energy' in r.lower() for r in rules_list) and metrics['exclamation_rate'] > 0.01:
        tips.append("Dial back exclamation and mirror the user's tone.")

    score = round(sim * 100)
    return {"score": score, "metrics": metrics, "suggestions": tips} 