from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from typing import List
from ..db import engine
from ..models.memory_event import MemoryEvent

router = APIRouter(tags=["memories"], prefix="/memories")

@router.post("/")
def create_memory(memory: MemoryEvent):
    with Session(engine) as ses:
        ses.add(memory)
        ses.commit()
        ses.refresh(memory)
    return memory

@router.get("/")
def list_memories() -> List[MemoryEvent]:
    with Session(engine) as ses:
        memories = ses.exec(select(MemoryEvent).order_by(MemoryEvent.timestamp.desc())).all()
    return memories

@router.get("/{event_id}")
def get_memory(event_id: str):
    with Session(engine) as ses:
        memory = ses.exec(select(MemoryEvent).where(MemoryEvent.event_id == event_id)).first()
        if not memory:
            raise HTTPException(404, "Memory not found")
    return memory 