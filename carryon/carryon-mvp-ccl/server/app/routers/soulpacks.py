from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from typing import List
import json
import os
from ..db import engine
from ..models.soulpack_meta import SoulpackMeta

router = APIRouter(tags=["soulpacks"], prefix="/soulpacks")

@router.post("/")
def load_soulpack(payload: dict):
    path = payload.get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(400, "Invalid path")
    
    try:
        with open(path, 'r') as f:
            content = f.read()
            data = json.loads(content)
        
        # Extract metadata
        name = data.get("name", "Unknown")
        version = data.get("version", "1.0.0")
        
        soulpack = SoulpackMeta(
            name=name,
            version=version,
            raw=content
        )
        
        with Session(engine) as ses:
            ses.add(soulpack)
            ses.commit()
            ses.refresh(soulpack)
        
        return {"ok": True, "id": soulpack.id, "name": name, "version": version}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to load soulpack: {str(e)}")

@router.get("/")
def list_soulpacks() -> List[SoulpackMeta]:
    with Session(engine) as ses:
        soulpacks = ses.exec(select(SoulpackMeta).order_by(SoulpackMeta.created_at.desc())).all()
    return soulpacks 