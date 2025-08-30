# config.py
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys - load from environment variables by default
    openweather_api_key: Optional[str] = Field(default=None, env="OPENWEATHER_API_KEY")
    arxiv_api_key: Optional[str] = Field(default=None, env="ARXIV_API_KEY")
    clinical_trials_api_key: Optional[str] = Field(default=None, env="CLINICAL_TRIALS_API_KEY")
    
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

# Global settings instance you import elsewhere
settings = Settings()


def apply_streamlit_secrets(st):
    """
    Override settings with Streamlit secrets at runtime.
    Call this from your Streamlit app AFTER import streamlit as st and AFTER Streamlit has initialized.
    Example (in main_app.py or your Streamlit entrypoint):
        import streamlit as st
        from config import settings, apply_streamlit_secrets
        apply_streamlit_secrets(st)

    This function will silently ignore missing secrets or errors.
    """
    try:
        if not hasattr(st, "secrets"):
            return

        # mapping from Streamlit secret keys to Settings attribute names
        key_map = {
            "OPENWEATHER_API_KEY": "openweather_api_key",
            "ARXIV_API_KEY": "arxiv_api_key",
            "CLINICAL_TRIALS_API_KEY": "clinical_trials_api_key",
        }

        for secret_key, attr_name in key_map.items():
            val = st.secrets.get(secret_key)
            if val:
                setattr(settings, attr_name, val)
    except Exception:
        # keep this function safe and non-failing during import/CI
        return
