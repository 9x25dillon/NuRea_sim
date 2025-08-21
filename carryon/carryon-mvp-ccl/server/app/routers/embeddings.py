from fastapi import APIRouter
from sqlmodel import Session, select
from ..db import engine
from ..models.memory_event import MemoryEvent
from ..retrieval.vector_index import rebuild_index, stats

router = APIRouter(tags=["embeddings"], prefix="/embeddings")

@router.post("/rebuild")
def rebuild():
    with Session(engine) as ses:
        mems = ses.exec(select(MemoryEvent)).all()
    return rebuild_index(mems)

@router.get("/stats")
def get_stats():
    return stats() 