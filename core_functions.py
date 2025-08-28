# core_functions.py
import os
import io
import base64
import time
import uuid
import gc
import re
import hashlib
import textwrap
import numpy as np
import pandas as pd
import psutil
import requests
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

# Optional imports with error handling
try:
    import torch
except ImportError:
    torch = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import arxiv
except ImportError:
    arxiv = None

from .config import settings

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
    def get_embedding_model(self):
        """Return SentenceTransformer model with progress tracking"""
        if not self.check_memory():
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
    def get_reranker(self):
        """Load reranker model with proper error handling"""
        if not self.check_memory():
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
    @st.cache_resource(show_spinner=False)
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
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            status_text.text("📄 Reading PDF structure...")
            progress_bar.progress(25)

            with open(tmp_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
                
                chunks = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    status_text.text(f"📖 Processing page {page_num + 1}/{total_pages}...")
                    progress_bar.progress(25 + (50 * page_num / total_pages))
                    
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
            progress_bar.progress(90)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            status_text.text("✅ PDF processing completed!")
            progress_bar.progress(100)
            
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
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📊 Reading CSV structure...")
            progress_bar.progress(25)
            
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            status_text.text("🔍 Analyzing data types...")
            progress_bar.progress(50)
            
            total_rows = len(df)
            
            status_text.text("📝 Creating searchable chunks...")
            progress_bar.progress(75)
            
            chunks = []
            for i, row in df.iterrows():
                if i % 100 == 0:
                    progress_bar.progress(75 + (20 * i / total_rows))
                
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
            progress_bar.progress(100)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return chunks

        except Exception as e:
            st.error(f"❌ CSV processing error: {str(e)}")
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
                
                if source_filter:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_fetch,
                        include=["metadatas", "documents", "distances"],
                        where={"source": source_filter}
                    )
                else:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_fetch,
                        include=["metadatas", "documents", "distances"],
                        where={}
                    )

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
                retrieved_docs["metadatas"][:3],
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

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            shard = data_type
            total_chunks = len(data)
            
            status_text.text(f"🔄 Preparing {total_chunks} chunks for ingestion...")
            progress_bar.progress(10)
            
            texts = []
            metadatas = []
            ids = []
            
            doc_id = hashlib.md5(str(source_name).encode('utf-8')).hexdigest()[:12]
            
            for i, chunk in enumerate(data):
                if i % 50 == 0:
                    progress_bar.progress(10 + (30 * i / total_chunks))
                
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
            progress_bar.progress(50)
            
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
                
                progress_bar.progress(50 + (40 * (i + len(batch_texts)) / len(texts)))
            
            status_text.text("💾 Storing in vector database...")
            progress_bar.progress(95)
            
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
            progress_bar.progress(100)
            
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
