from typing import Any, Dict, List
from .entropy_engine import entropy_engine
from .matrix_processor import matrix_processor
from .al_uls import al_uls
from .motif_engine import motif_engine
from .suggestions import SUGGESTIONS

def _prefix_match(prefix: str, state: str) -> List[str]:
    pre = (prefix or "").upper()
    pool = SUGGESTIONS.get(state, [])
    return [t for t in pool if t.startswith(pre)]

def _apply_token_to_qgi(qgi: Dict[str, Any], token_text: str) -> None:
    entropy_score = entropy_engine.score_token(token_text)
    volatility_signal = entropy_engine.get_volatility_signal(token_text)
    qgi.setdefault("entropy_scores", []).append(entropy_score)
    qgi["volatility"] = volatility_signal
    if al_uls.is_symbolic_call(token_text):
        symbolic_func = al_uls.parse_symbolic_call(token_text)
        qgi.setdefault("symbolic_calls", []).append(symbolic_func)
    tags = motif_engine.detect_tags(token_text)
    if tags:
        existing = set(qgi.get("motif_tags", []))
        for t in tags:
            if t not in existing:
                qgi.setdefault("motif_tags", []).append(t)
                existing.add(t)

async def _apply_token_to_qgi_async(qgi: Dict[str, Any], token_text: str) -> None:
    _apply_token_to_qgi(qgi, token_text)
    if qgi.get("symbolic_calls"):
        last = qgi["symbolic_calls"][-1]
        res = await al_uls.eval_symbolic_call_async(last)
        qgi.setdefault("symbolic_results", []).append(res)

def api_suggest(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    _apply_token_to_qgi(qgi, prefix)
    if use_semantic and matrix_processor.available():
        suggestions = matrix_processor.semantic_state_suggest(prefix, state)
    else:
        suggestions = _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}

async def api_suggest_async(prefix: str = "", state: str = "S0", use_semantic: bool = True) -> Dict[str, Any]:
    qgi: Dict[str, Any] = {
        "state": state,
        "prefix": prefix,
        "selects": [],
        "filters": [],
        "group_by": [],
        "order": None,
        "tokens": [],
        "entropy_scores": [],
        "volatility": None,
        "symbolic_calls": [],
        "symbolic_results": [],
        "retrieval_routes": [],
        "motif_tags": []
    }
    qgi["tokens"].append(prefix)
    await _apply_token_to_qgi_async(qgi, prefix)
    if use_semantic and matrix_processor.available():
        suggestions = matrix_processor.semantic_state_suggest(prefix, state)
    else:
        suggestions = _prefix_match(prefix, state)
    return {"suggestions": suggestions, "qgi": qgi}