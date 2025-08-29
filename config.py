# config.py
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
import os

class Settings(BaseSettings):
    # API Keys
    openweather_api_key: str = st.secrets["OPENWEATHER_API_KEY"]
    arxiv_api_key: Optional[str] = None
    clinical_trials_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    max_chunk_size: int = 800
    max_results: int = 5
    similarity_threshold: float = 0.3
    
    # Database Configuration
    chroma_db_path: str = "./nextgen_rag_db"
    domain_shards: List[str] = ["pdf", "csv"]
    
    # Performance Configuration
    memory_threshold: int = 85
    batch_size: int = 32
    auto_refresh_interval: int = 300  # seconds
    
    # UI Configuration
    default_city: str = "Delhi"
    max_chat_history: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
