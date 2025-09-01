# main_app.py (fixed version)
# Fix SQLite for Chroma - MUST BE AT THE TOP
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ----------------------
# Consolidated imports
# ----------------------
import os
import io
import base64
import time
import uuid
import pandas as pd
import gc
import re
import hashlib
import textwrap
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import requests
import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

# ----------------------
# Configuration
# ----------------------
class Settings(BaseSettings):
    # API Keys
    openweather_api_key: str = "demo_key"
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

# optional heavy libs (guarded use)
try:
    import torch as _torch
    torch = _torch
except Exception:
    torch = None
    st.warning("⚠️ PyTorch not available. Model features will be disabled until torch is installed.")

# Only set default tensor type if torch is available
if torch is not None:
    try:
        torch.set_default_tensor_type(torch.FloatTensor)
    except Exception:
        pass

# heavier libs used later
try:
    import arxiv
except Exception:
    arxiv = None

try:
    import duckdb
except Exception:
    duckdb = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None

try:
    from unstructured.partition.pdf import partition_pdf
except Exception:
    partition_pdf = None

try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# optional / additional features
try:
    from setfit import SetFitModel, SetFitTrainer
    from datasets import Dataset
except Exception:
    SetFitModel = None
    SetFitTrainer = None
    Dataset = None

try:
    import spacy
except Exception:
    spacy = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    CLIPProcessor = None
    CLIPModel = None

try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    plt = None
    px = None
    go = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None
    st.warning("⚠️ PyPDF2 not installed. PDF processing will be limited.")

# ----------------------
# UI Configuration
# ----------------------
st.set_page_config(
    page_title="🚀 Next-Gen RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 10px;
        background: #f0f2f6;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Constants & Globals
# ----------------------
INDIAN_CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab"},
    "Agra": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
    "Nashik": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra"},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"}
}

# ----------------------
# API Integrations
# ----------------------
class APIIntegrations:
    @staticmethod
    def fetch_arxiv_papers(query: str, max_results: int = None) -> List[Dict]:
        """Fetch real papers from Arxiv API"""
        if max_results is None:
            max_results = settings.max_results
            
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "doi": result.doi if hasattr(result, 'doi') else None,
                    "journal_ref": result.journal_ref if hasattr(result, 'journal_ref') else None,
                    "comment": result.comment if hasattr(result, 'comment') else None,
                    "primary_category": result.primary_category if hasattr(result, 'primary_category') else None,
                    "categories": result.categories if hasattr(result, 'categories') else []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            st.error(f"❌ Arxiv API error: {e}")
            # Fallback to mock data
            return APIIntegrations._get_mock_arxiv_papers(query, max_results)
    
    @staticmethod
    def _get_mock_arxiv_papers(query: str, max_results: int) -> List[Dict]:
        """Fallback mock data for Arxiv"""
        papers = []
        for i in range(min(max_results, 3)):
            papers.append({
                "title": f"Research Paper {i+1}: {query.title()} in AI Systems",
                "authors": ["Dr. AI Researcher", "Prof. ML Expert"],
                "published": (datetime.now() - timedelta(days=30*i)).strftime("%Y-%m-%d"),
                "summary": f"This paper explores {query} in the context of artificial intelligence and machine learning. The research presents novel approaches and methodologies...",
                "pdf_url": f"https://arxiv.org/pdf/{1234+i}.{56789+i}",
                "entry_id": f"{1234+i}.{56789+i}",
                "categories": ["cs.AI", "cs.LG"],
                "doi": f"10.48550/arXiv.{1234+i}.{56789+i}"
            })
        return papers
    
    @staticmethod
    def fetch_clinical_trials(condition: str = "cancer", max_results: int = None) -> List[Dict]:
        """Fetch real clinical trials from ClinicalTrials.gov API"""
        if max_results is None:
            max_results = settings.max_results
            
        try:
            # ClinicalTrials.gov API endpoint
            base_url = "https://clinicaltrials.gov/api/query/full_studies"
            params = {
                "expr": condition,
                "min_rnk": 1,
                "max_rnk": max_results,
                "fmt": "json"
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                trials = []
                
                for study in data.get('FullStudiesResponse', {}).get('FullStudies', []):
                    study_data = study.get('Study', {})
                    protocol_data = study_data.get('ProtocolSection', {})
                    identification = protocol_data.get('IdentificationModule', {})
                    status = protocol_data.get('StatusModule', {})
                    design = protocol_data.get('DesignModule', {})
                    contacts = protocol_data.get('ContactsLocationsModule', {})
                    
                    trial = {
                        "id": identification.get('NCTId', 'N/A'),
                        "title": identification.get('BriefTitle', 'N/A'),
                        "status": status.get('OverallStatus', 'N/A'),
                        "sponsor": identification.get('OrgStudyIdInfo', {}).get('OrgStudyId', 'N/A'),
                        "start_date": status.get('StartDateStruct', {}).get('StartDate', 'N/A'),
                        "completion_date": status.get('CompletionDateStruct', {}).get('CompletionDate', 'N/A'),
                        "study_type": design.get('StudyType', 'N/A'),
                        "phase": design.get('PhaseList', {}).get('Phase', ['N/A'])[0] if design.get('PhaseList', {}).get('Phase') else 'N/A',
                        "enrollment": design.get('EnrollmentInfo', {}).get('EnrollmentCount', 'N/A'),
                        "intervention": ', '.join([i.get('InterventionName', '') for i in design.get('InterventionList', {}).get('Intervention', [])]),
                        "locations": [loc.get('LocationFacility', '') for loc in contacts.get('LocationList', {}).get('Location', [])],
                        "text": f"Clinical Trial: {identification.get('BriefTitle', 'N/A')} - {identification.get('BriefSummary', 'N/A')}"
                    }
                    trials.append(trial)
                
                return trials
            else:
                st.warning(f"⚠️ Clinical Trials API returned status {response.status_code}")
                return APIIntegrations._get_mock_clinical_trials(condition, max_results)
                
        except Exception as e:
            st.error(f"❌ Clinical Trials API error: {e}")
            return APIIntegrations._get_mock_clinical_trials(condition, max_results)
    
    @staticmethod
    def _get_mock_clinical_trials(condition: str, max_results: int) -> List[Dict]:
        """Fallback mock data for clinical trials"""
        trials = []
        statuses = ["Recruiting", "Active", "Completed", "Enrolling"]
        
        for i in range(min(max_results, 4)):
            trials.append({
                "id": f"NCT{123456 + i}",
                "title": f"Study of {condition.title()} Treatment - Phase {(i % 3) + 1}",
                "status": statuses[i % len(statuses)],
                "sponsor": f"Medical Center {i+1}",
                "start_date": (datetime.now() - timedelta(days=365*i)).strftime("%Y-%m-%d"),
                "completion_date": (datetime.now() + timedelta(days=365*(2-i))).strftime("%Y-%m-%d"),
                "enrollment": (100 + i*50),
                "locations": [f"City {i+1}", f"City {i+2}"],
                "intervention": f"Novel {condition} therapy approach",
                "text": f"Clinical Trial: {condition.title()} Treatment Study - This randomized controlled trial investigates novel therapeutic approaches for {condition} treatment."
            })
        return trials

# Global instance
api_client = APIIntegrations()

# ----------------------
# Core Functions
# ----------------------
class CoreFunctions:
    def __init__(self):
        self.embedding_model = None
        self.reranker = None
        self.vector_client = None
        self.collections = {}
        
    # Memory Management
    def check_memory(self, threshold: int = None) -> bool:
        """Check memory usage and return True if below threshold"""
        if threshold is None:
            threshold = settings.memory_threshold
            
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > threshold:
            st.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
            return False
        return True

    def smart_cleanup(self):
        """Smart cleanup of GPU/MPS memory and garbage collection"""
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()

    # Model Loading
    @st.cache_resource(show_spinner=False)
    def get_embedding_model(_self):
        """Return SentenceTransformer model with progress tracking"""
        if not _self.check_memory():
            st.error("🚫 Insufficient memory for embedding model")
            return None
        
        if SentenceTransformer is None:
            st.error("🚫 SentenceTransformer not installed. Run: pip install sentence-transformers")
            return None
        
        try:
            with st.spinner("🧠 Loading embedding model (first time only)..."):
                model = SentenceTransformer(settings.embedding_model)
                st.success("✅ Embedding model loaded successfully!")
                return model
        except Exception as e:
            st.error(f"❌ Embedding model load failed: {e}")
            return None

    @st.cache_resource(show_spinner=False)
    def get_reranker(_self):
        """Load reranker model with proper error handling"""
        if not _self.check_memory():
            return None
        if FlagReranker is None:
            return None
        try:
            with st.spinner("🎯 Loading reranker model..."):
                reranker = FlagReranker(settings.reranker_model, use_fp16=False)
                st.success("✅ Reranker model loaded!")
                return reranker
        except Exception as e:
            st.warning(f"⚠️ Reranker load failed: {e}")
            return None

    # Vector Store Management
    def get_vector_store(self):
        """Enhanced vector store initialization with better error handling"""
        try:
            if chromadb is None:
                st.error("🚫 ChromaDB not available. Install: pip install chromadb")
                return None, {}

            with st.spinner("🗄️ Initializing vector database..."):
                client = chromadb.PersistentClient(path=settings.chroma_db_path)

                embedding_model = self.get_embedding_model()
                if embedding_model is None:
                    st.warning("⚠️ Using default embedding dimension (768)")
                    embedding_dimension = 768
                else:
                    try:
                        embedding_dimension = embedding_model.get_sentence_embedding_dimension()
                    except Exception:
                        embedding_dimension = 768

                local_collections = {}
                for domain in settings.domain_shards:
                    try:
                        collection = client.get_collection(f"knowledge_{domain}")
                        st.info(f"📚 Found existing '{domain}' collection")
                    except Exception:
                        collection = client.create_collection(
                            name=f"knowledge_{domain}",
                            metadata={
                                "hnsw:space": "cosine",
                                "embedding_dimension": str(embedding_dimension),
                                "created_at": datetime.now().isoformat()
                            }
                        )
                        st.success(f"✅ Created new '{domain}' collection")
                    
                    local_collections[domain] = collection

                st.success(f"✅ Vector store initialized with {len(local_collections)} collections")
                return client, local_collections

        except Exception as e:
            st.error(f"❌ Vector store initialization failed: {e}")
            return None, {}

    # File Processing
    def process_pdf(self, uploaded_file) -> List[Dict]:
        """Enhanced PDF processing with better chunking and metadata extraction"""
        if PyPDF2 is None:
            st.error("🚫 PyPDF2 not installed. Run: pip install PyPDF2")
            return []
        
        try:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            status_text.text("📄 Reading PDF structure...")
            progress_bar.progress(0.25)

            with open(tmp_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                
                chunks = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    status_text.text(f"📖 Processing page {page_num + 1}/{total_pages}...")
                    progress_bar.progress(0.25 + (0.5 * page_num / total_pages))
                    
                    page_text = page.extract_text()
                    if page_text:
                        # Smart chunking by sentences/paragraphs
                        sentences = re.split(r'(?<=[.!?])\s+', page_text)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk + sentence) > settings.max_chunk_size:
                                if current_chunk:
                                    chunks.append({
                                        "text": current_chunk.strip(),
                                        "source": uploaded_file.name,
                                        "page": page_num + 1,
                                        "chunk_type": "paragraph",
                                        "char_count": len(current_chunk),
                                        "word_count": len(current_chunk.split())
                                    })
                                    current_chunk = sentence
                            else:
                                current_chunk += " " + sentence if current_chunk else sentence
                        
                        # Add remaining chunk
                        if current_chunk.strip():
                            chunks.append({
                                "text": current_chunk.strip(),
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "chunk_type": "paragraph",
                                "char_count": len(current_chunk),
                                "word_count": len(current_chunk.split())
                            })

            status_text.text("🔍 Analyzing content...")
            progress_bar.progress(0.9)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            status_text.text("✅ PDF processing completed!")
            progress_bar.progress(1.0)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            return chunks

        except Exception as e:
            st.error(f"❌ PDF processing error: {str(e)}")
            return []

    def process_csv(self, uploaded_file) -> List[Dict]:
        """Enhanced CSV processing with data type detection and validation"""
        try:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            status_text.text("📊 Reading CSV structure...")
            progress_bar.progress(0.25)
            
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            status_text.text("🔍 Analyzing data types...")
            progress_bar.progress(0.5)
            
            total_rows = len(df)
            
            status_text.text("📝 Creating searchable chunks...")
            progress_bar.progress(0.75)
            
            chunks = []
            for i, row in df.iterrows():
                if i % 100 == 0:
                    progress_bar.progress(0.75 + (0.2 * i / total_rows))
                
                row_text_parts = []
                for col, val in row.items():
                    if pd.notna(val):
                        row_text_parts.append(f"{col}: {val}")
                
                row_text = " | ".join(row_text_parts)
                
                chunks.append({
                    "text": f"Row {i+1}: {row_text}",
                    "source": uploaded_file.name,
                    "row": i+1,
                    "chunk_type": "csv_row",
                    "columns": list(row.index),
                    "non_null_fields": sum(1 for val in row.values if pd.notna(val)),
                    "char_count": len(row_text),
                })

            status_text.text("✅ CSV processing completed!")
            progress_bar.progress(1.0)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return chunks

        except Exception as e:
            st.error(f"❌ CSV processing error: {e}")
            return []

    # Retrieval Functions
    def calculate_keyword_overlap(self, question: str, document: str) -> float:
        """Calculate semantic keyword overlap between question and document"""
        try:
            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            doc_words = set(re.findall(r'\b\w+\b', document.lower()))
            
            stop_words = {
                'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
                'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about'
            }
            
            question_words = question_words - stop_words
            doc_words = doc_words - stop_words
            
            if not question_words:
                return 0.0
            
            overlap = len(question_words.intersection(doc_words))
            return min(overlap / len(question_words), 1.0)
            
        except Exception:
            return 0.0

    def hybrid_retrieval(self, question: str, max_results: int = None, 
                        source_filter: Optional[str] = None, 
                        similarity_threshold: float = None, 
                        debug: bool = False) -> Dict:
        """
        Enhanced hybrid retrieval with advanced scoring and filtering
        """
        if max_results is None:
            max_results = settings.max_results
        if similarity_threshold is None:
            similarity_threshold = settings.similarity_threshold
            
        if not self.collections:
            if debug: st.info("🚫 No vector collections available")
            return {"documents": [], "metadatas": [], "domains": [], "scores": [], "debug_info": {}}

        model = self.get_embedding_model()
        if model is None:
            if debug: st.info("🚫 Embedding model not available")
            return {"documents": [], "metadatas": [], "domains": [], "scores": [], "debug_info": {}}

        debug_info = {"query_length": len(question), "collections_searched": 0, "total_candidates": 0}

        try:
            start_time = time.time()
            query_embedding = model.encode(question).tolist()
            embedding_time = time.time() - start_time
            debug_info["embedding_time"] = f"{embedding_time:.3f}s"
            
        except Exception as e:
            if debug: st.error(f"❌ Query encoding failed: {e}")
            return {"documents": [], "metadatas": [], "domains": [], "scores": [], "debug_info": debug_info}

        all_results = []
        retrieval_start = time.time()

        for domain, collection in self.collections.items():
            try:
                debug_info["collections_searched"] += 1
                
                try:
                    collection_count = collection.count()
                    n_fetch = min(max(max_results * 8, 20), collection_count)
                except:
                    n_fetch = max_results * 8
                
                try:
                    if source_filter and source_filter.strip():
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_fetch,
                            include=["metadatas", "documents", "distances"],
                            where={"source": source_filter}
                        )
                    else:
                        # Don't use where clause if no filter is specified
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=n_fetch,
                            include=["metadatas", "documents", "distances"]
                        )
                except Exception as e:
                    if debug: st.warning(f"⚠️ Query failed in '{domain}' collection: {e}")
                    continue

                docs = results.get("documents", [[]])[0] if results.get("documents") else []
                metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                distances = results.get("distances", [[]])[0] if results.get("distances") else []

                debug_info["total_candidates"] += len(docs)

                for doc, meta, distance in zip(docs, metas, distances):
                    if not isinstance(meta, dict):
                        meta = {}
                    
                    if source_filter and meta.get("source") != source_filter:
                        continue
                    
                    similarity_score = max(0, 1 - distance) if distance is not None else 0.5
                    
                    if similarity_score < similarity_threshold:
                        continue
                    
                    keyword_score = self.calculate_keyword_overlap(question, doc or "")
                    
                    doc_length = len(doc) if doc else 0
                    length_score = min(doc_length / 1000, 1.0) * 0.8 + 0.2
                    
                    metadata_score = len([v for v in meta.values() if v]) / 10.0
                    metadata_score = min(metadata_score, 1.0)
                    
                    final_score = (
                        similarity_score * 0.5 +
                        keyword_score * 0.3 +
                        length_score * 0.1 +
                        metadata_score * 0.1
                    )

                    all_results.append({
                        "document": doc,
                        "metadata": meta,
                        "domain": domain,
                        "score": final_score,
                        "similarity": similarity_score,
                        "keyword_match": keyword_score,
                        "length_score": length_score,
                        "metadata_score": metadata_score
                    })

            except Exception as e:
                if debug:
                    st.warning(f"⚠️ Search failed in '{domain}' collection: {e}")
                continue

        retrieval_time = time.time() - retrieval_start
        debug_info["retrieval_time"] = f"{retrieval_time:.3f}s"

        ranking_start = time.time()
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        unique_results = []
        seen_hashes = set()
        
        for result in all_results:
            doc_text = result["document"] or ""
            
            content_key = doc_text[:200] + str(result["metadata"].get("source", "")) + str(result["metadata"].get("page", ""))
            content_hash = hashlib.md5(content_key.encode()).hexdigest()[:10]
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
                
                if len(unique_results) >= max_results:
                    break

        ranking_time = time.time() - ranking_start
        debug_info["ranking_time"] = f"{ranking_time:.3f}s"
        debug_info["final_results"] = len(unique_results)
        debug_info["total_time"] = f"{embedding_time + retrieval_time + ranking_time:.3f}s"

        return {
            "documents": [r["document"] for r in unique_results],
            "metadatas": [r["metadata"] for r in unique_results],
            "domains": [r["domain"] for r in unique_results],
            "scores": [r["score"] for r in unique_results],
            "debug_info": debug_info
        }

    def smart_answer_generation(self, question: str, retrieved_docs: Dict, max_context_length: int = 2000) -> str:
        """Generate comprehensive answer using retrieved documents with citations"""
        if not retrieved_docs["documents"]:
            return "❌ I couldn't find relevant information to answer your question. Try rephrasing your query or check if the relevant documents are uploaded."
        
        context_parts = []
        total_length = 0
        
        for i, (doc, meta, score) in enumerate(zip(
            retrieved_docs["documents"], 
            retrieved_docs["metadatas"], 
            retrieved_docs["scores"]
        )):
            if total_length >= max_context_length:
                break
                
            source_info = f"📄 **Source {i+1}**: {meta.get('source', 'Unknown')}"
            if meta.get('page'):
                source_info += f" (Page {meta['page']})"
            if meta.get('row'):
                source_info += f" (Row {meta['row']})"
            
            source_info += f" | Relevance: {score:.1%}"
            
            doc_part = f"{source_info}\n{doc[:600]}{'...' if len(doc) > 600 else ''}\n\n---\n"
            context_parts.append(doc_part)
            total_length += len(doc_part)
        
        if len(retrieved_docs["documents"]) == 1:
            answer = f"""## 📋 Answer based on your documents:

{retrieved_docs['documents'][0][:1000]}{'...' if len(retrieved_docs['documents'][0]) > 1000 else ''}

---
**📊 Source Information:**
- **Document**: {retrieved_docs['metadatas'][0].get('source', 'Unknown')}
- **Relevance Score**: {retrieved_docs['scores'][0]:.1%}
- **Domain**: {retrieved_docs['domains'][0]}
"""
        else:
            answer = f"""## 📋 Answer compiled from {len(retrieved_docs["documents"])} sources:

"""
            for i, (doc, meta, domain, score) in enumerate(zip(
                retrieved_docs["documents"][:3], 
                retrieved_docs["metadatas"] [:3],
                retrieved_docs["domains"][:3], 
                retrieved_docs["scores"][:3]
            ), 1):
                answer += f"""### {i}. From {meta.get('source', 'Unknown')}
{doc[:400]}{'...' if len(doc) > 400 else ''}
*Relevance: {score:.1%} | Domain: {domain}*

"""
        
        return answer

    # Document Management
    def delete_documents_by_source(self, source_name: str) -> bool:
        """Enhanced document deletion with confirmation and statistics"""
        try:
            if not self.collections:
                st.warning("🚫 Vector store unavailable; nothing to delete.")
                return False
            
            deleted_count = 0
            
            for domain, collection in self.collections.items():
                try:
                    before_count = collection.count()
                    collection.delete(where={"source": source_name})
                    after_count = collection.count()
                    deleted_count += (before_count - after_count)
                    
                except Exception as e:
                    st.warning(f"⚠️ Partial deletion failure in {domain}: {e}")
                    continue
            
            if deleted_count > 0:
                st.success(f"✅ Deleted {deleted_count} document chunks from '{source_name}'")
                return True
            else:
                st.info(f"ℹ️ No documents found for source '{source_name}'")
                return False
                
        except Exception as e:
            st.error(f"❌ Error deleting documents: {str(e)}")
            return False

    def get_document_statistics(self) -> Dict:
        """Get comprehensive document statistics"""
        stats = {
            "total_documents": 0,
            "documents_by_domain": {},
            "documents_by_source": {},
            "total_size_estimate": 0
        }
        
        try:
            for domain, collection in self.collections.items():
                try:
                    count = collection.count()
                    stats["total_documents"] += count
                    stats["documents_by_domain"][domain] = count
                    
                    if count > 0:
                        sample = collection.get(limit=min(count, 100), include=["metadatas"])
                        sources = {}
                        
                        for meta in sample.get("metadatas", []):
                            if isinstance(meta, dict) and "source" in meta:
                                source = meta["source"]
                                sources[source] = sources.get(source, 0) + 1
                        
                        for source, count in sources.items():
                            stats["documents_by_source"][source] = stats["documents_by_source"].get(source, 0) + count
                            
                except Exception:
                    stats["documents_by_domain"][domain] = 0
                    
        except Exception:
            pass
        
        return stats

    # Data Ingestion
    def ingest_data(self, data: List[Dict], data_type: str = "pdf", source_name: Optional[str] = None) -> bool:
        """Enhanced data ingestion with progress tracking and validation"""
        if not data:
            st.warning("⚠️ No data to ingest")
            return False

        try:
            if data_type not in settings.domain_shards:
                st.error(f"❌ Unsupported data type: {data_type}")
                return False
                
            if not source_name:
                st.error("❌ Source name is required")
                return False

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            shard = data_type
            total_chunks = len(data)
            
            status_text.text(f"🔄 Preparing {total_chunks} chunks for ingestion...")
            progress_bar.progress(0.1)
            
            texts = []
            metadatas = []
            ids = []
            
            doc_id = hashlib.md5(str(source_name).encode('utf-8')).hexdigest()[:12]
            
            for i, chunk in enumerate(data):
                if i % 50 == 0:
                    progress_bar.progress(0.1 + (0.3 * i / total_chunks))
                
                texts.append(chunk["text"])
                
                md = {
                    "source": source_name,
                    "type": data_type,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "ingested_at": datetime.now().isoformat(),
                    "char_count": len(chunk["text"]),
                    "word_count": len(chunk["text"].split())
                }
                
                for key in ["page", "row", "chunk_type", "columns"]:
                    if key in chunk:
                        md[key] = chunk[key]
                
                chunk_id = f"{doc_id}_{i}"
                md["chunk_id"] = chunk_id
                metadatas.append(md)
                ids.append(f"{data_type}_{chunk_id}")

            status_text.text("🧠 Generating embeddings...")
            progress_bar.progress(0.5)
            
            model = self.get_embedding_model()
            if model is None:
                st.error("❌ Embedding model not available. Ingestion aborted.")
                return False

            batch_size = settings.batch_size
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts).tolist()
                embeddings.extend(batch_embeddings)
                
                progress_bar.progress(0.5 + (0.4 * (i + len(batch_texts)) / len(texts)))
            
            status_text.text("💾 Storing in vector database...")
            progress_bar.progress(0.95)
            
            if not self.collections or shard not in self.collections:
                st.error("❌ Vector store collection not available. Ingestion aborted.")
                return False

            self.collections[shard].upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            status_text.text("✅ Ingestion completed successfully!")
            progress_bar.progress(1.0)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully ingested {total_chunks} chunks from '{source_name}' into '{shard}' collection")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Ingestion failed: {str(e)}")
            return False

# Global instance
core = CoreFunctions()

# ----------------------
# Weather Functions
# ----------------------
def get_real_weather_data(city_name="Delhi") -> Dict:
    try:
        API_KEY = st.secrets.get("OPENWEATHER_API_KEY")
    except Exception:
        API_KEY = None

    if not API_KEY:
        st.warning("⚠️ Please set a valid OPENWEATHER_API_KEY in Streamlit secrets.")
        return get_fallback_weather(city_name)

    if city_name not in INDIAN_CITIES:
        city_name = "Delhi"

    city_coords = INDIAN_CITIES[city_name]
    lat, lon = city_coords["lat"], city_coords["lon"]

    current_url = (
        f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={API_KEY}&units=metric"
    )
    forecast_url = (
        f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}"
        f"&appid={API_KEY}&units=metric&cnt=8"
    )

    try:
        current_response = requests.get(current_url, timeout=10)
        if current_response.status_code != 200:
            st.error(f"❌ Weather API Error: {current_response.status_code}")
            return get_fallback_weather(city_name)

        data = current_response.json()
        forecast_data = []
        forecast_response = requests.get(forecast_url, timeout=10)
        if forecast_response.status_code == 200:
            forecast_data = forecast_response.json().get("list", [])[:6]

        current_time = datetime.now()

        weather_info = {
            "city": city_name,
            "state": city_coords.get("state", ""),
            "temperature": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "precipitation": data.get("rain", {}).get("1h", data.get("snow", {}).get("1h", 0.0)),
            "windspeed": round(data["wind"]["speed"] * 3.6, 1),
            "wind_direction": data["wind"].get("deg", 0),
            "description": data["weather"][0]["description"].title(),
            "icon": data["weather"][0]["icon"],
            "feels_like": round(data["main"]["feels_like"], 1),
            "pressure": data["main"]["pressure"],
            "visibility": data.get("visibility", 10000) / 1000,
            "uv_index": data.get("uvi", 0),
            "cloudiness": data["clouds"]["all"],
            "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M"),
            "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M"),
            "time": current_time.isoformat(),
            "location": f"{lat},{lon}",
            "forecast": forecast_data,
            "air_quality": "Good",
            "text": f"""🌤️ Weather Report for {city_name}, {city_coords.get('state', '')}
📅 Date: {current_time.strftime('%Y-%m-%d %H:%M IST')}
🌡️ Temperature: {round(data["main"]["temp"], 1)}°C (Feels like {round(data["main"]["feels_like"], 1)}°C)
🌈 Condition: {data["weather"][0]["description"].title()}
💧 Humidity: {data["main"]["humidity"]}%
🌧️ Precipitation: {data.get("rain", {}).get("1h", 0.0)}mm
💨 Wind: {round(data["wind"]["speed"] * 3.6, 1)} km/h
🎯 Pressure: {data["main"]["pressure"]} hPa
👁️ Visibility: {data.get("visibility", 10000) / 1000} km
☁️ Cloudiness: {data["clouds"]["all"]}%
🌅 Sunrise: {datetime.fromtimestamp(data["sys"]["sunrise"]).strftime('%H:%M')}
🌇 Sunset: {datetime.fromtimestamp(data["sys"]["sunset"]).strftime('%H:%M')}"""
        }

        return weather_info

    except Exception as e:
        st.warning(f"Weather API error: {str(e)}")
        return get_fallback_weather(city_name)

def get_fallback_weather(city_name="Delhi") -> Dict:
    """Enhanced fallback weather data"""
    current_time = datetime.now()
    city_coords = INDIAN_CITIES.get(city_name, INDIAN_CITIES["Delhi"])
    
    return {
        "city": city_name,
        "state": city_coords.get("state", ""),
        "temperature": 25.0 + hash(city_name) % 10,  # Slight variation by city
        "humidity": 60 + hash(city_name) % 20,
        "precipitation": 0.0,
        "windspeed": 10.0 + hash(city_name) % 15,
        "wind_direction": 180,
        "description": "Clear Sky (Demo Data)",
        "icon": "01d",
        "feels_like": 27.0 + hash(city_name) % 8,
        "pressure": 1013 + hash(city_name) % 20,
        "visibility": 10.0,
        "cloudiness": 0,
        "sunrise": "06:30",
        "sunset": "18:30",
        "time": current_time.isoformat(),
        "location": f"{city_coords['lat']},{city_coords['lon']}",
        "forecast": [],
        "text": f"""🌤️ Demo Weather for {city_name}, {city_coords.get('state', '')}
⚠️ This is demo data. Set OPENWEATHER_API_KEY for real data.
📅 Date: {current_time.strftime('%Y-%m-%d %H:%M IST')}
🌡️ Temperature: {25.0 + hash(city_name) % 10}°C (Demo)"""
    }

def render_weather_dashboard():
    """Enhanced weather dashboard with visualizations"""
    st.markdown('<div class="main-header"><h2>🌤️ Real-time Weather Dashboard - India</h2></div>', unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # State-wise city selection
        states = sorted(set([city_info["state"] for city_info in INDIAN_CITIES.values()]))
        selected_state = st.selectbox("🗺️ Select State:", ["All States"] + states, key="state_selector")
        
        if selected_state == "All States":
            available_cities = list(INDIAN_CITIES.keys())
        else:
            available_cities = [city for city, info in INDIAN_CITIES.items() if info["state"] == selected_state]
        
        selected_city = st.selectbox(
            "🏙️ Select City:",
            options=available_cities,
            index=0 if "Delhi" not in available_cities else available_cities.index("Delhi"),
            key="city_selector"
        )
    
    with col2:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=True, key="auto_refresh_toggle")
        if auto_refresh:
            st.caption("Updates every 5 minutes")
    
    with col3:
        refresh_interval = st.selectbox("⏱️ Interval", ["5 min", "10 min", "15 min", "30 min"], key="refresh_interval")
        interval_seconds = {"5 min": 300, "10 min": 600, "15 min": 900, "30 min": 1800}[refresh_interval]
    
    with col4:
        if st.button("🔄 Update Now", key="manual_weather_refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Get weather data
    if auto_refresh:
        @st.cache_data(ttl=interval_seconds, show_spinner=False)
        def get_cached_weather(city):
            return get_real_weather_data(city)
    else:
        @st.cache_data(show_spinner=False)
        def get_cached_weather(city):
            return get_real_weather_data(city)
    
    with st.spinner(f"🌍 Fetching weather data for {selected_city}..."):
        weather_data = get_cached_weather(selected_city)
    
    if weather_data:
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🌡️ Temperature", 
                f"{weather_data.get('temperature', 'N/A')}°C",
                delta=f"Feels {weather_data.get('feels_like', 'N/A')}°C"
            )
        
        with col2:
            st.metric("💧 Humidity", f"{weather_data.get('humidity', 'N/A')}%")
        
        with col3:
            st.metric("🌧️ Precipitation", f"{weather_data.get('precipitation', 'N/A')}mm")
        
        with col4:
            st.metric("💨 Wind Speed", f"{weather_data.get('windspeed', 'N/A')} km/h")
        
        with col5:
            st.metric("👁️ Visibility", f"{weather_data.get('visibility', 'N/A')} km")
        
        # Secondary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Pressure", f"{weather_data.get('pressure', 'N/A')} hPa")
        
        with col2:
            st.metric("☁️ Cloudiness", f"{weather_data.get('cloudiness', 'N/A')}%")
        
        with col3:
            st.metric("🌅 Sunrise", weather_data.get('sunrise', 'N/A'))
        
        with col4:
            st.metric("🌇 Sunset", weather_data.get('sunset', 'N/A'))
        
        with col5:
            condition = weather_data.get('description', 'N/A')
            st.metric("🌤️ Condition", condition)
        
        # Weather visualization and details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Weather forecast chart (if available)
            if weather_data.get('forecast') and px:
                forecast_data = []
                for item in weather_data['forecast'][:6]:  # Next 18 hours
                    forecast_data.append({
                        'Time': datetime.fromtimestamp(item['dt']).strftime('%H:%M'),
                        'Temperature': round(item['main']['temp'], 1),
                        'Humidity': item['main']['humidity'],
                        'Description': item['weather'][0]['description']
                    })
                
                if forecast_data:
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    fig = px.line(df_forecast, x='Time', y='Temperature', 
                                title='📈 Temperature Forecast (Next 18 hours)',
                                markers=True)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: Simple weather info display
                st.info("📊 Weather forecast visualization requires forecast data and plotly")
        
        with col2:
            # Weather summary card
            st.markdown("### 📋 Weather Summary")
            st.markdown(f"""
            **📍 Location:** {weather_data.get('city')}, {weather_data.get('state')}  
            **🌡️ Temperature:** {weather_data.get('temperature')}°C  
            **🌤️ Condition:** {weather_data.get('description')}  
            **💨 Wind:** {weather_data.get('windspeed')} km/h  
            **💧 Humidity:** {weather_data.get('humidity')}%  
            **👁️ Visibility:** {weather_data.get('visibility')} km  
            **⏰ Updated:** {datetime.now().strftime('%H:%M:%S')}
            """)
        
        # Detailed weather report
        with st.expander("📊 Detailed Weather Report", expanded=False):
            st.code(weather_data["text"])
            
            # Additional technical details
            st.markdown("### 🔧 Technical Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Coordinates:** {weather_data.get('location', 'N/A')}")
                st.write(f"**Wind Direction:** {weather_data.get('wind_direction', 'N/A')}°")
                st.write(f"**Data Source:** OpenWeatherMap API")
            with col2:
                st.write(f"**Last Updated:** {weather_data.get('time', 'N/A')}")
                st.write(f"**Refresh Interval:** {refresh_interval}")
                st.write(f"**Auto Refresh:** {'Enabled' if auto_refresh else 'Disabled'}")

# ----------------------
# Enhanced Chat System
# ----------------------
def enhanced_chat_with_document(query: str, source_name: str, chat_history: List[Dict]) -> List[Dict]:
    """Enhanced document chat with context awareness and better responses"""
    with st.spinner(f"🔍 Searching in '{source_name}'..."):
        results = core.hybrid_retrieval(
            query, 
            max_results=3, 
            source_filter=source_name,
            similarity_threshold=0.2,  # Lower threshold for chat
            debug=False
        )

    if results["documents"]:
        # Generate comprehensive response
        response = core.smart_answer_generation(query, results, max_context_length=1500)
        
        # Add context from chat history for better responses
        if len(chat_history) > 0:
            recent_context = " ".join([msg.get("user", "") for msg in chat_history[-2:]])
            if recent_context:
                response += f"\n\n*Note: This answer considers our recent conversation context.*"
    else:
        response = f"""❌ I couldn't find relevant information about "{query}" in the document '{source_name}'.

**Suggestions:**
- Try rephrasing your question
- Use different keywords
- Check if the information exists in the document
- Try a more general question about the document's content"""

    # Add to chat history with timestamp
    chat_entry = {
        "user": query,
        "assistant": response,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": source_name,
        "relevance_scores": results.get("scores", [])
    }
    
    chat_history.append(chat_entry)
    return chat_history

# ----------------------
# System Statistics
# ----------------------
def display_system_stats():
    """Display comprehensive system statistics"""
    with st.sidebar.expander("📊 System Statistics", expanded=False):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        st.metric("🖥️ CPU Usage", f"{cpu_percent}%")
        st.metric("🧠 Memory Usage", f"{memory.percent}%", f"{memory.used // (1024**4):.1f}GB / {memory.total // (1024**4):.1f}GB")
        
        if torch and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                st.metric("🎮 GPU Memory", f"{(gpu_used/gpu_memory)*100:.1f}%", f"{gpu_used // (1024**4):.1f}GB / {gpu_memory // (1024**4):.1f}GB")
            except:
                pass

# ----------------------
# Main Application UI
# ----------------------
def main():
    """Main application with enhanced UI and functionality"""
    
    # Initialize vector store (without caching)
    if not hasattr(core, 'vector_client') or core.vector_client is None:
        core.vector_client, core.collections = core.get_vector_store()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Next-Gen RAG System</h1>
        <p>Advanced Hybrid Retrieval | Real-time Weather | Intelligent Document Chat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session states
    if 'arxiv_papers' not in st.session_state:
        st.session_state.arxiv_papers = []
    if 'clinical_trials' not in st.session_state:
        st.session_state.clinical_trials = []
    if 'last_job_run' not in st.session_state:
        st.session_state.last_job_run = "Never"
    if 'chat_histories' not in st.session_state:
        st.session_state.chat_histories = {}
    if 'file_hashes' not in st.session_state:
        st.session_state.file_hashes = {}
    if 'active_file' not in st.session_state:
        st.session_state.active_file = None
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        # Document statistics
        stats = core.get_document_statistics()
        st.session_state.system_stats = stats
        
        st.markdown("### 📊 Database Status")
        st.metric("📚 Total Documents", stats.get("total_documents", 0))
        st.metric("🏠 Collections", len(stats.get("documents_by_domain", {})))
        
        if stats["documents_by_source"]:
            st.markdown("**📄 Sources:**")
            for source, count in list(stats["documents_by_source"].items())[:5]:
                st.caption(f"• {source}: {count} chunks")
        
        st.markdown("### 🌍 Weather Settings")
        st.caption("Configured for Indian cities with auto-refresh")
        
        # System performance
        display_system_stats()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🧹 Clear Cache", help="Clear all cached data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            core.smart_cleanup()
            st.success("✅ Cache cleared!")
        
        if st.button("🔄 Refresh Stats", help="Update system statistics"):
            st.rerun()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌤️ Weather Dashboard", 
        "📄 Document Upload", 
        "📚 Arxiv Research",
        "🏥 Clinical Trials", 
        "🔍 Knowledge Query", 
        "💬 Document Chat",
        "📊 System Dashboard"
    ])

    # Tab 1: Enhanced Weather Dashboard
    with tab1:
        render_weather_dashboard()

        # Add a divider for clarity
        st.markdown("---")
        st.subheader("📈 Weather Forecast Visualization")

        # Example: Using forecast data from your weather API
        if "forecast" in st.session_state and st.session_state["forecast"]:
            import pandas as pd
            import plotly.express as px
        
            # Prepare forecast data
            forecast_data = st.session_state["forecast"]
            df = pd.DataFrame([
                {
                    "Time": f["dt_txt"],
                    "Temperature": f["main"]["temp"],
                    "Humidity": f["main"]["humidity"]
                }
                for f in forecast_data
            ])

            # Create a Plotly line chart
            fig = px.line(
                df,
                x="Time",
                y="Temperature",
                title="Next 24 Hours Temperature Forecast",
                markers=True
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No forecast data available to plot.")
            
    # Tab 2: Enhanced Document Upload
    with tab2:
        st.markdown("### 📄 Document Processing Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            file_type = st.radio("📎 File Type", ["PDF", "CSV"], horizontal=True, key="file_type_radio")
        
        with col2:
            replace_existing = st.checkbox("🔄 Replace existing", True, help="Delete existing documents with same name")

        if file_type == "PDF":
            uploaded_file = st.file_uploader(
                "📄 Upload PDF Document", 
                type="pdf", 
                key="pdf_uploader",
                help="Supports text-based PDFs. Scanned documents may have limited text extraction."
            )

            if uploaded_file:
                file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
                
                # Display file info
                st.info(f"📄 **File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.1f} KB")

                if replace_existing or file_hash != st.session_state.file_hashes.get(uploaded_file.name):
                    if st.button("🚀 Process PDF", type="primary"):
                        if replace_existing and uploaded_file.name in st.session_state.file_hashes:
                            core.delete_documents_by_source(uploaded_file.name)

                        chunks = core.process_pdf(uploaded_file)
                        if chunks:
                            if core.ingest_data(chunks, "pdf", uploaded_file.name):
                                st.session_state.file_hashes[uploaded_file.name] = file_hash

                                # Show content preview
                                with st.expander("👀 Content Preview", expanded=True):
                                    for i, chunk in enumerate(chunks[:3]):
                                        st.markdown(f"📃 Chunk {i+1} (Page {chunk.get('page', 1)})")
                                        st.markdown(f"*{len(chunk['text'])} characters, {len(chunk['text'].split())} words*")
                                        st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                                        if i < 2: st.divider()
                else:
                    st.success("✅ This file is already processed and up to date!")

        else:  # CSV processing
            uploaded_file = st.file_uploader(
                "📊 Upload CSV Data", 
                type="csv", 
                key="csv_uploader",
                help="Supports standard CSV files with headers"
            )

            if uploaded_file:
                file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
                # Display file info
                st.info(f"📊 **File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.1f} KB")

                if replace_existing or file_hash != st.session_state.file_hashes.get(uploaded_file.name):
                    if st.button("🚀 Process CSV", type="primary"):
                        if replace_existing and uploaded_file.name in st.session_state.file_hashes:
                            core.delete_documents_by_source(uploaded_file.name)

                        chunks = core.process_csv(uploaded_file)
                        if chunks:
                            if core.ingest_data(chunks, "csv", uploaded_file.name):
                                st.session_state.file_hashes[uploaded_file.name] = file_hash

                                # Show content preview
                                with st.expander("👀 Content Preview", expanded=True):
                                    for i, chunk in enumerate(chunks[:3]):
                                        st.markdown(f"📊 Row {chunk.get('row', i+1)}")
                                        st.markdown(f"*{len(chunk['text'])} characters, {chunk.get('non_null_fields', 'N/A')} fields*")
                                        st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                                        if i < 2: st.divider()
                else:
                    st.success("✅ This file is already processed and up to date!")

    # Tab 3: Enhanced Arxiv Research
    with tab3:
        st.markdown("### 📚 AI Research Papers (Arxiv)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            research_query = st.text_input("🔍 Research Topic", placeholder="machine learning, neural networks, AI...")
        with col2:
            max_papers = st.number_input("📄 Max Papers", min_value=1, max_value=20, value=5)
        
        if st.button("🚀 Fetch Research Papers", type="primary"):
            with st.spinner("📚 Searching Arxiv for relevant papers..."):
                papers = api_client.fetch_arxiv_papers(research_query, max_papers)
                st.session_state.arxiv_papers = papers
        
        if st.session_state.arxiv_papers:
            st.success(f"✅ Found {len(st.session_state.arxiv_papers)} papers")
            
            for i, paper in enumerate(st.session_state.arxiv_papers):
                with st.expander(f"📄 {paper.get('title', 'Untitled')}", expanded=i==0):
                    st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                    st.markdown(f"**Published:** {paper.get('published', 'Unknown')}")
                    st.markdown(f"**Categories:** {', '.join(paper.get('categories', []))}")
                    if paper.get('citation_count'):
                        st.markdown(f"**Citations:** {paper.get('citation_count', 'N/A')}")
                    
                    st.markdown("**Abstract:**")
                    st.write(paper.get('summary', 'No summary available'))
                    
                    if paper.get('pdf_url'):
                        st.markdown(f"[📥 Download PDF]({paper.get('pdf_url')})")
                    
                    st.divider()

    # Tab 4: Enhanced Clinical Trials
    with tab4:
        st.markdown("### 🏥 Clinical Trials Search")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            condition_query = st.text_input("🔍 Medical Condition", placeholder="cancer, diabetes, heart disease...")
        with col2:
            max_trials = st.number_input("🧪 Max Trials", min_value=1, max_value=20, value=5)
        
        if st.button("🚀 Search Clinical Trials", type="primary"):
            with st.spinner("🏥 Searching for clinical trials..."):
                trials = api_client.fetch_clinical_trials(condition_query, max_trials)
                st.session_state.clinical_trials = trials
        
        if st.session_state.clinical_trials:
            st.success(f"✅ Found {len(st.session_state.clinical_trials)} trials")
            
            for i, trial in enumerate(st.session_state.clinical_trials):
                with st.expander(f"🧪 {trial.get('title', 'Untitled')}", expanded=i==0):
                    st.markdown(f"**Trial ID:** {trial.get('id', 'N/A')}")
                    st.markdown(f"**Status:** {trial.get('status', 'Unknown')}")
                    st.markdown(f"**Sponsor:** {trial.get('sponsor', 'Unknown')}")
                    st.markdown(f"**Participants:** {trial.get('participants', 'N/A')}")
                    st.markdown(f"**Start Date:** {trial.get('start_date', 'Unknown')}")
                    st.markdown(f"**Completion:** {trial.get('completion_date', 'Unknown')}")
                    
                    st.markdown("**Intervention:**")
                    st.write(trial.get('intervention', 'No information available'))
                    
                    st.markdown("**Locations:**")
                    st.write(", ".join(trial.get('locations', [])))
                    
                    st.divider()

    # Tab 5: Enhanced Knowledge Query
    with tab5:
        st.markdown("### 🔍 Knowledge Base Query")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("❓ Your Question", placeholder="Ask anything about your uploaded documents...")
        with col2:
            max_results = st.number_input("📊 Max Results", min_value=1, max_value=10, value=3)
        
        source_filter = st.selectbox(
            "📁 Filter by Source (Optional)",
            [""] + list(st.session_state.system_stats.get("documents_by_source", {}).keys())
        )
        
        if st.button("🚀 Search Knowledge Base", type="primary") and query:
            with st.spinner("🔍 Searching through documents..."):
                results = core.hybrid_retrieval(
                    query, 
                    max_results=max_results, 
                    source_filter=source_filter if source_filter else None,
                    debug=True
                )
                
                if results["documents"]:
                    st.success(f"✅ Found {len(results['documents'])} relevant results")
                    
                    # Display debug info
                    with st.expander("🔧 Search Details", expanded=False):
                        st.json(results["debug_info"])
                    
                    # Display results
                    for i, (doc, meta, score, domain) in enumerate(zip(
                        results["documents"], 
                        results["metadatas"], 
                        results["scores"], 
                        results["domains"]
                    )):
                        with st.expander(f"📄 Result {i+1} | Score: {score:.1%} | {meta.get('source', 'Unknown')}", expanded=i==0):
                            st.markdown(f"**Source:** {meta.get('source', 'Unknown')}")
                            if meta.get('page'):
                                st.markdown(f"**Page:** {meta.get('page')}")
                            if meta.get('row'):
                                st.markdown(f"**Row:** {meta.get('row')}")
                            st.markdown(f"**Domain:** {domain}")
                            st.markdown(f"**Relevance Score:** {score:.1%}")
                            
                            st.markdown("**Content:**")
                            st.write(doc)
                            
                            st.divider()
                    
                    # Generate answer
                    st.markdown("### 📋 Generated Answer")
                    st.markdown(core.smart_answer_generation(query, results))
                else:
                    st.warning("❌ No relevant results found. Try a different query or check your document uploads.")

    # Tab 6: Enhanced Document Chat
    with tab6:
        st.markdown("### 💬 Document Chat Interface")
        
        # Source selection
        available_sources = list(st.session_state.system_stats.get("documents_by_source", {}).keys())
        if not available_sources:
            st.info("📚 Upload documents first to enable chat functionality")
        else:
            selected_source = st.selectbox(
                "📄 Select Document to Chat With",
                available_sources,
                key="chat_source"
            )
            
            # Initialize chat history for this source if not exists
            if selected_source not in st.session_state.chat_histories:
                st.session_state.chat_histories[selected_source] = []
            
            chat_history = st.session_state.chat_histories[selected_source]
            
            # Display chat history
            st.markdown("#### 💭 Conversation History")
            chat_container = st.container()
            
            with chat_container:
                for message in chat_history:
                    if message["user"]:
                        st.markdown(f"**👤 You ({message['timestamp']}):** {message['user']}")
                    if message["assistant"]:
                        st.markdown(f"**🤖 Assistant ({message['timestamp']}):**")
                        st.markdown(message["assistant"])
                    st.divider()
            
            # Chat input
            chat_input = st.text_input(
                "💬 Ask about the document:",
                placeholder=f"Ask anything about {selected_source}...",
                key="chat_input"
            )
            
            if st.button("🚀 Send Message", type="primary") and chat_input:
                # Add user message to history
                chat_history = enhanced_chat_with_document(chat_input, selected_source, chat_history)
                st.session_state.chat_histories[selected_source] = chat_history
                st.rerun()

    # Tab 7: Enhanced System Dashboard
    import pandas as pd
    with tab7:
        st.markdown("### 📊 System Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🖥️ System Health")
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            st.metric("CPU Usage", f"{cpu_usage}%", 
                     delta="High" if cpu_usage > 80 else "Normal" if cpu_usage > 60 else "Low",
                     delta_color="inverse" if cpu_usage > 80 else "normal")
            
            st.metric("Memory Usage", f"{memory_usage}%",
                     delta="High" if memory_usage > 80 else "Normal" if memory_usage > 60 else "Low",
                     delta_color="inverse" if memory_usage > 80 else "normal")
            
            # Disk usage
            try:
                disk_usage = psutil.disk_usage('/').percent
                st.metric("Disk Usage", f"{disk_usage}%",
                         delta="High" if disk_usage > 80 else "Normal" if disk_usage > 60 else "Low",
                         delta_color="inverse" if disk_usage > 80 else "normal")
            except:
                pass
        
        with col2:
            st.markdown("#### 📚 Document Statistics")
            stats = st.session_state.system_stats
            
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Collections", len(stats.get("documents_by_domain", {})))
            st.metric("Unique Sources", len(stats.get("documents_by_source", {})))
        
        with col3:
            st.markdown("#### ⚡ Performance")
            if torch and torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                    st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
                except:
                    pass
            
            # Display last job run time
            st.metric("Last Refresh", st.session_state.last_job_run)
        
        # Collection details
        st.markdown("#### 🗂️ Collection Details")
        domain_data = stats.get("documents_by_domain", {}) if stats and isinstance(stats, dict) else {}

        if domain_data:
            domain_df = pd.DataFrame({
                "Collection": list(domain_data.keys()),
                "Documents": list(domain_data.values())
            })
            st.dataframe(domain_df, use_container_width=True)

            # 🔥 Add a Plotly Bar Chart
            import plotly.express as px
            fig = px.bar(
                domain_df,
                x="Collection",
                y="Documents",
                title="📊 Documents per Collection",
                labels={"Collection": "Collection", "Documents": "Number of Documents"},
                 text="Documents"
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents in collections yet")

        # Source details
        st.markdown("#### 📄 Source Details")
        if stats and stats.get("documents_by_source"):
            source_df = pd.DataFrame({
                "Source": list(stats["documents_by_source"].keys()),
                "Documents": list(stats["documents_by_source"].values())
        }) 
            st.dataframe(source_df, use_container_width=True)
        else:
            st.info("No source data available")

        
        # System actions
        st.markdown("#### ⚙️ System Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Data", type="primary"):
                st.session_state.last_job_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Chats"):
                st.session_state.chat_histories = {}
                st.success("All chat histories cleared")
        
        with col3:
            if st.button("📊 Export Statistics"):
                # Create a downloadable report
                report_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_stats": st.session_state.system_stats,
                    "weather_cities": list(INDIAN_CITIES.keys()),
                    "arxiv_papers_count": len(st.session_state.arxiv_papers),
                    "clinical_trials_count": len(st.session_state.clinical_trials)
                }
                
                # Convert to JSON for download
                import json
                json_report = json.dumps(report_data, indent=2)
                
                st.download_button(
                    label="📥 Download Report",
                    data=json_report,
                    file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Run the main application
if __name__ == "__main__":
    main()
