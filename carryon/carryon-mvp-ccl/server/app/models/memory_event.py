from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
import uuid

class MemoryEvent(SQLModel, table=True):
    __tablename__ = "memory_events"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), unique=True, index=True)
    data: str = Field(description="Memory content")
    subject: str = Field(description="Subject/topic of the memory")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[str] = Field(default=None, description="JSON metadata")
    
    @property
    def recency_score(self) -> float:
        """Calculate recency score based on timestamp"""
        now = datetime.utcnow()
        age_hours = (now - self.timestamp).total_seconds() / 3600
        # Exponential decay: newer = higher score
        return max(0.1, 1.0 / (1.0 + age_hours / 24.0)) 