"""
Knowledge Base Utilities

Simple utilities for creating Redis-based knowledge bases from text content.
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple

import redis
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redisvl.index import SearchIndex
import numpy as np
from sentence_transformers import SentenceTransformer
# Configure logging
logger = logging.getLogger("kb-utils")

class FreeTextVectorizer:
    """Free embedding vectorizer using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dims = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {model_name} with {self.dims} dimensions")
    
    def embed(self, text: str, as_buffer: bool = True):
        embedding = self.model.encode(text, convert_to_numpy=True)
        if as_buffer:
            return embedding.astype(np.float32).tobytes()
        return embedding
    
class KnowledgeBaseManager:
    """Manages Redis-based knowledge bases for web content"""
    
    def __init__(self, redis_client: redis.Redis, embeddings: Optional[Any] = None):
        self.redis_client = redis_client
        self.embeddings = embeddings or FreeTextVectorizer()
        self.embedding_dims = getattr(self.embeddings, 'dims', 384)
        self.active_indexes = {}
        logger.info(f"Using embeddings with {self.embedding_dims} dimensions")
    
    def create_knowledge_base(
        self, 
        source_id: str, 
        content, 
        chunk_size: int = 2500, 
        chunk_overlap: int = 250,
        skip_chunking: bool = False
    ) -> Tuple[bool, str, Optional[SearchIndex]]:
        """
        Create a knowledge base from content.
        
        Args:
            source_id: Identifier for the content source (URL or custom ID)
            content: Either a string (will be chunked) or list of strings (used as-is)
            chunk_size: Size of text chunks (ignored if content is list or skip_chunking=True)
            chunk_overlap: Overlap between chunks (ignored if content is list or skip_chunking=True)
            skip_chunking: If True, treat string content as single chunk
            
        Returns:
            Tuple of (success, message, search_index)
        """
        try:
            if not content:
                return False, "No content to process", None
            
            # Create unique index name based on source ID
            source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
            index_name = f"kb-{source_hash}"
            
            # Handle different content types
            if isinstance(content, list):
                # Content is already a list of text chunks
                text_chunks = content
                source_type = "text_list"
                logger.info(f"Using provided list of {len(text_chunks)} text chunks")
            elif isinstance(content, str):
                if skip_chunking:
                    # Treat the entire string as one chunk
                    text_chunks = [content]
                    source_type = "text_single"
                    logger.info("Using entire text as single chunk")
                else:
                    # Split content into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Create documents and split
                    docs = [Document(page_content=content, metadata={"source_id": source_id, "source": "text"})]
                    doc_chunks = splitter.split_documents(docs)
                    text_chunks = [chunk.page_content for chunk in doc_chunks]
                    source_type = "text_chunked"
                    logger.info(f"Split text into {len(text_chunks)} chunks")
            else:
                return False, f"Unsupported content type: {type(content)}", None
            
            # Create Redis search index schema
            schema = {
                "index": {"name": index_name, "prefix": f"kb:{source_hash}:"},
                "fields": [
                    {"name": "content", "type": "text"},
                    {"name": "source_id", "type": "tag"},
                    {"name": "source_type", "type": "tag"},
                    {"name": "chunk_index", "type": "numeric"},
                    {
                        "name": "content_vector",
                        "type": "vector",
                        "attrs": {
                            "dims": self.embedding_dims,  # Dynamic dimensions
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32",
                        },
                    },
                ],
            }
            
            # Create and populate knowledge base
            kb_index = SearchIndex.from_dict(schema, redis_client=self.redis_client)
            kb_index.create(overwrite=True)
            
            # Prepare documents for indexing
            payload = []
            for i, text_chunk in enumerate(text_chunks):
                try:
                    embedding = self.embeddings.embed(text_chunk, as_buffer=True)
                    payload.append({
                        "content": text_chunk,
                        "source_id": source_id,
                        "source_type": source_type,
                        "chunk_index": i,
                        "content_vector": embedding,
                    })
                except Exception as e:
                    logger.warning(f"Failed to embed chunk {i}: {e}")
                    continue
            
            if not payload:
                return False, "Failed to create embeddings for content", None
            
            # Load data into the index
            kb_index.load(payload)
            
            # Store reference to active index
            self.active_indexes[source_hash] = {
                "index": kb_index,
                "source_id": source_id,
                "source_type": source_type,
                "chunks": len(text_chunks),
                "created_at": time.time()
            }
            
            success_msg = f"✅ Created knowledge base with {len(text_chunks)} chunks ({source_type})"
            logger.info(success_msg)
            return True, success_msg, kb_index
            
        except Exception as e:
            error_msg = f"❌ Knowledge base creation failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def get_index_for_source(self, source_id: str) -> Optional[SearchIndex]:
        """Get the search index for a specific source ID"""
        source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
        index_info = self.active_indexes.get(source_hash)
        return index_info["index"] if index_info else None
    
    # Backward compatibility
    def get_index_for_url(self, url: str) -> Optional[SearchIndex]:
        """Get the search index for a specific URL (backward compatibility)"""
        return self.get_index_for_source(url)
    
    def clear_knowledge_base(self, source_id: str = None) -> str:
        """
        Clear knowledge base(s).
        
        Args:
            source_id: Specific source ID to clear (clears all if None)
            
        Returns:
            Status message
        """
        try:
            if source_id:
                # Clear specific source
                source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
                if source_hash in self.active_indexes:
                    index_info = self.active_indexes[source_hash]
                    index_info["index"].drop()
                    del self.active_indexes[source_hash]
                    return f"✅ Cleared knowledge base for {source_id}"
                else:
                    return f"⚠️ No knowledge base found for {source_id}"
            else:
                # Clear all indexes
                cleared_count = 0
                for source_hash, index_info in list(self.active_indexes.items()):
                    try:
                        index_info["index"].drop()
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clear index {source_hash}: {e}")
                
                self.active_indexes.clear()
                return f"✅ Cleared {cleared_count} knowledge bases"
                
        except Exception as e:
            error_msg = f"❌ Clear operation failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all active knowledge bases"""
        status = {
            "total_indexes": len(self.active_indexes),
            "indexes": []
        }
        
        for source_hash, index_info in self.active_indexes.items():
            status["indexes"].append({
                "source_hash": source_hash,
                "source_id": index_info["source_id"],
                "source_type": index_info["source_type"],
                "chunks": index_info["chunks"],
                "created_at": index_info["created_at"],
                "age_seconds": time.time() - index_info["created_at"]
            })
        
        return status




def create_knowledge_base_from_texts(
    texts: List[str],
    source_id: str = "custom_texts",
    redis_url: str = "redis://localhost:6379",
    skip_chunking: bool = True,
    embeddings: Optional[Any] = None
) -> Tuple[bool, str, Optional[SearchIndex]]:
    """
    Convenience function to create a knowledge base directly from texts.
    
    Args:
        texts: List of text strings
        source_id: Identifier for the content source
        redis_url: Redis connection URL
        skip_chunking: If True, use texts as-is; if False, chunk each text
        
    Returns:
        Tuple of (success, message, search_index)
    """
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    kb_manager = KnowledgeBaseManager(redis_client,embeddings=embeddings)
    return kb_manager.create_knowledge_base(source_id, texts, skip_chunking=skip_chunking)


