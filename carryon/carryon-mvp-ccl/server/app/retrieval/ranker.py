from typing import List, Dict
from ..models.memory_event import MemoryEvent
from .vector_index import search
from .graph_store import degree
from ..config import settings

def rank_memories(query: str, mems: List[MemoryEvent], k: int = 12, entropy: float = 0.0) -> List[MemoryEvent]:
    # Preselect via vector search
    id2mem: Dict[str, MemoryEvent] = {m.event_id: m for m in mems}
    hits = search(query, top_n=max(k*4, 32))
    hit_ids = [hid for hid, _ in hits]
    candidates = [id2mem[h] for h in hit_ids if h in id2mem] or mems

    deg_map = degree([m.subject for m in candidates])
    # Build a dict of base sem scores from hits
    sem_map = {hid: score for hid, score in hits}
    scored = []
    for m in candidates:
        s = sem_map.get(m.event_id, 0.0)
        rec = m.recency_score
        gd = deg_map.get(m.subject, 0.0)
        score = settings.alpha*s + settings.beta*rec + settings.gamma*gd + settings.delta*entropy
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    k_eff = max(4, int(k * (0.75 if entropy > 0.5 else 1.0)))
    return [m for _, m in scored[:k_eff]] 