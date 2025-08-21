from fastapi import APIRouter, Body, HTTPException
from sqlmodel import Session, select
from ..db import engine
from ..models.soulpack_meta import SoulpackMeta
from ..tone.alignment import evaluate_tone
import json

router = APIRouter(tags=["tone"], prefix="/tone")

@router.post("/evaluate")
def evaluate(payload: dict = Body(...)):
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(400, "text is required")
    # Load latest soulpack
    with Session(engine) as ses:
        row = ses.exec(select(SoulpackMeta).order_by(SoulpackMeta.id.desc())).first()
    if not row:
        raise HTTPException(404, "No soulpack available")
    sp = json.loads(row.raw)
    voice = sp.get("persona", {}).get("voice", {})
    return evaluate_tone(voice, text) 