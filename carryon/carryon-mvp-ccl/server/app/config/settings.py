from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Vector index paths
    faiss_index_path: str = "./data/faiss_index"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Ranking weights
    alpha: float = 0.4      # Semantic similarity weight
    beta: float = 0.3       # Recency weight  
    gamma: float = 0.2      # Graph centrality weight
    delta: float = 0.1      # Entropy adjustment weight
    
    # Database
    database_url: str = "sqlite:///./carryon.db"
    
    class Config:
        env_file = ".env"

settings = Settings() 