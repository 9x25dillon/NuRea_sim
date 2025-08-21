from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class SoulpackMeta(SQLModel, table=True):
    __tablename__ = "soulpack_meta"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(description="Soulpack name")
    version: str = Field(description="Soulpack version")
    raw: str = Field(description="Raw JSON content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow) 