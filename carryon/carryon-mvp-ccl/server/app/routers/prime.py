from fastapi import APIRouter, Body
from sqlmodel import Session, select
from typing import List
from ..db import engine
from ..models.memory_event import MemoryEvent
from ..retrieval.ranker import rank_memories

router = APIRouter(tags=["prime"], prefix="/prime")

@router.post("/")
def prime_context(payload: dict = Body(...)):
    query = payload.get("query", "")
    if not query.strip():
        return {"error": "Query is required"}
    
    # Get relevant memories for priming
    with Session(engine) as ses:
        memories = ses.exec(select(MemoryEvent)).all()
    
    if not memories:
        return {"primed": False, "reason": "No memories available"}
    
    # Rank memories by relevance to query
    ranked = rank_memories(query, memories, k=8)
    
    return {
        "primed": True,
        "query": query,
        "context_count": len(ranked),
        "context": [{"id": m.event_id, "data": m.data[:200] + "..." if len(m.data) > 200 else m.data} for m in ranked]
    }

@router.post("/query")
def query_with_context(payload: dict = Body(...)):
    query = payload.get("query", "")
    if not query.strip():
        return {"error": "Query is required"}
    
    # Get relevant memories
    with Session(engine) as ses:
        memories = ses.exec(select(MemoryEvent)).all()
    
    if not memories:
        return {"error": "No memories available"}
    
    # Rank and return top memories
    ranked = rank_memories(query, memories, k=12)
    
    return {
        "query": query,
        "results": [{"id": m.event_id, "data": m.data, "subject": m.subject, "timestamp": m.timestamp.isoformat()} for m in ranked]
    } 