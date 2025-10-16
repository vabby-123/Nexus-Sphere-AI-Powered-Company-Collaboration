"""
Production-grade vector store with ChromaDB (no Docker) and FAISS support
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging
from pathlib import Path
import json
import pickle

import numpy as np
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store (NO DOCKER REQUIRED)
    
    Features:
    - Persistent local storage
    - Fast similarity search
    - Metadata filtering
    - No external dependencies
    - Perfect for development and production
    """
    
    def __init__(
        self,
        collection_name: str = "nexus_sphere_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            embedding_model: Gemini embedding model
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"✅ Initialized ChromaDB at {self.persist_directory}")
    
    async def add_documents_batch(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        Add documents in batches
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents per batch
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        total_added = 0
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Adding {len(documents)} documents in {num_batches} batches")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract texts and metadatas
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Generate unique IDs
                ids = [
                    hashlib.md5(f"{doc.metadata.get('doc_id', '')}_{doc.page_content[:100]}".encode()).hexdigest()
                    for doc in batch
                ]
                
                # Add to ChromaDB
                await asyncio.to_thread(
                    self.vectorstore.add_texts,
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch)
                logger.info(f"Batch {i//batch_size + 1}/{num_batches}: Added {len(batch)} documents")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
        
        # Persist changes
        await asyncio.to_thread(self.vectorstore.persist)
        
        logger.info(f"✅ Successfully added {total_added} documents")
        
        return total_added
    
    def search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search
        
        Args:
            query: Search query
            filter_dict: Metadata filters
            limit: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build filter
            chroma_filter = self._build_chroma_filter(filter_dict) if filter_dict else None
            
            # Perform search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=limit,
                filter=chroma_filter
            )
            
            # Format results
            formatted_results = []
            
            for doc, score in results:
                # Convert distance to similarity (ChromaDB uses L2 distance)
                # Lower distance = higher similarity
                similarity_score = 1 / (1 + score)
                
                # Apply score threshold if specified
                if score_threshold and similarity_score < score_threshold:
                    continue
                
                formatted_results.append({
                    'id': doc.metadata.get('doc_id', 'unknown'),
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': similarity_score
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _build_chroma_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB filter from dictionary
        
        ChromaDB filter format:
        {"field": "value"} for exact match
        {"field": {"$in": [val1, val2]}} for multiple values
        """
        chroma_filter = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                # Multiple values (OR condition)
                chroma_filter[key] = {"$in": value}
            elif isinstance(value, dict):
                # Range query
                if 'gte' in value:
                    chroma_filter[key] = {"$gte": value['gte']}
                elif 'lte' in value:
                    chroma_filter[key] = {"$lte": value['lte']}
            else:
                # Single value (exact match)
                chroma_filter[key] = value
        
        return chroma_filter
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents matching filter"""
        try:
            chroma_filter = self._build_chroma_filter(filter_dict)
            
            # Get matching documents
            collection = self.vectorstore._collection
            results = collection.get(where=chroma_filter)
            
            if results['ids']:
                # Delete by IDs
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents")
                return len(results['ids'])
            
            return 0
        
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                'name': self.collection_name,
                'points_count': count,
                'persist_directory': str(self.persist_directory),
                'embedding_model': 'Gemini embedding-001'
            }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def scroll_points(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict], Optional[int]]:
        """
        Scroll through collection points
        
        Returns:
            Tuple of (points, next_offset)
        """
        try:
            collection = self.vectorstore._collection
            
            chroma_filter = self._build_chroma_filter(filter_dict) if filter_dict else None
            
            results = collection.get(
                where=chroma_filter,
                limit=limit,
                offset=offset,
                include=['documents', 'metadatas']
            )
            
            formatted_points = []
            
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    formatted_points.append({
                        'id': doc_id,
                        'payload': {
                            'text': results['documents'][i],
                            **results['metadatas'][i]
                        }
                    })
            
            # Calculate next offset
            next_offset = offset + len(formatted_points) if len(formatted_points) == limit else None
            
            return formatted_points, next_offset
        
        except Exception as e:
            logger.error(f"Scroll error: {e}")
            return [], None


class FAISSVectorStore:
    """
    FAISS-based vector store (In-memory with disk persistence)
    
    Features:
    - Lightning fast similarity search
    - Lower memory footprint
    - Perfect for < 1M documents
    - Can save/load from disk
    """
    
    def __init__(
        self,
        index_path: str = "./faiss_index",
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize FAISS vector store
        
        Args:
            index_path: Path to save/load FAISS index
            embedding_model: Gemini embedding model
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Try to load existing index
        self.vectorstore = self._load_or_create_index()
        
        logger.info(f"✅ Initialized FAISS at {self.index_path}")
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_file = self.index_path / "index.faiss"
        
        if index_file.exists():
            try:
                vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for FAISS
                )
                logger.info("Loaded existing FAISS index")
                return vectorstore
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}")
        
        # Create new empty index
        # Initialize with a dummy document
        dummy_doc = Document(
            page_content="Initialization document",
            metadata={"type": "init"}
        )
        
        vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        
        logger.info("Created new FAISS index")
        return vectorstore
    
    async def add_documents_batch(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        Add documents in batches
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents per batch
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Add to FAISS
                await asyncio.to_thread(
                    self.vectorstore.add_documents,
                    batch
                )
                
                total_added += len(batch)
                logger.info(f"Added batch: {len(batch)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
        
        # Save index to disk
        await self._save_index()
        
        logger.info(f"✅ Successfully added {total_added} documents")
        
        return total_added
    
    async def _save_index(self):
        """Save FAISS index to disk"""
        try:
            await asyncio.to_thread(
                self.vectorstore.save_local,
                str(self.index_path)
            )
            logger.info("Saved FAISS index to disk")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search
        
        Note: FAISS doesn't support metadata filtering natively,
        so we'll filter results after retrieval
        """
        try:
            # Get more results than needed (for post-filtering)
            fetch_limit = limit * 3 if filter_dict else limit
            
            # Perform search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_limit
            )
            
            # Format and filter results
            formatted_results = []
            
            for doc, score in results:
                # Convert distance to similarity
                similarity_score = 1 / (1 + score)
                
                # Apply score threshold
                if score_threshold and similarity_score < score_threshold:
                    continue
                
                # Apply metadata filter
                if filter_dict:
                    if not self._matches_filter(doc.metadata, filter_dict):
                        continue
                
                formatted_results.append({
                    'id': doc.metadata.get('doc_id', 'unknown'),
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': similarity_score
                })
                
                # Stop if we have enough results
                if len(formatted_results) >= limit:
                    break
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range query
                if 'gte' in value and metadata[key] < value['gte']:
                    return False
                if 'lte' in value and metadata[key] > value['lte']:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            # FAISS doesn't track document count directly
            # We can estimate from index
            index = self.vectorstore.index
            count = index.ntotal if hasattr(index, 'ntotal') else 0
            
            return {
                'name': 'faiss_index',
                'points_count': count,
                'index_path': str(self.index_path),
                'embedding_model': 'Gemini embedding-001'
            }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """
        Delete documents matching filter
        
        Note: FAISS doesn't support deletion, so we rebuild the index
        """
        logger.warning("FAISS doesn't support deletion. Index rebuild required.")
        return 0


class HybridSearchEngine:
    """
    Hybrid search combining vector and keyword search
    
    Works with both ChromaDB and FAISS
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Initialize BM25 (optional)
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_available = True
            self.bm25_index = None  # Will be built on-demand
        except ImportError:
            logger.warning("BM25 not available, hybrid search will use vector-only")
            self.bm25_available = False
    
    def hybrid_search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and keyword methods
        
        Args:
            query: Search query
            filter_dict: Metadata filters
            limit: Number of results
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            Ranked search results
        """
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        vector_weight /= total_weight
        keyword_weight /= total_weight
        
        # Vector search
        vector_results = self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            limit=limit * 2  # Get more candidates
        )
        
        # If BM25 not available, return vector results
        if not self.bm25_available:
            return vector_results[:limit]
        
        # Combine scores
        combined_results = []
        
        for result in vector_results:
            # Vector score (already normalized 0-1)
            vector_score = result['score']
            
            # Keyword score (simple term matching)
            keyword_score = self._calculate_keyword_score(query, result['text'])
            
            # Combined score
            combined_score = (vector_weight * vector_score + 
                            keyword_weight * keyword_score)
            
            result['combined_score'] = combined_score
            result['vector_score'] = vector_score
            result['keyword_score'] = keyword_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results[:limit]
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Simple keyword scoring"""
        query_terms = set(query.lower().split())
        text_lower = text.lower()
        
        matches = sum(1 for term in query_terms if term in text_lower)
        
        return matches / len(query_terms) if query_terms else 0.0


# ============================================================================
# FACTORY FUNCTION - Choose your vector store
# ============================================================================

def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "nexus_sphere_knowledge",
    persist_directory: str = "./vector_db"
):
    """
    Factory function to create vector store
    
    Args:
        store_type: "chroma" or "faiss"
        collection_name: Collection name
        persist_directory: Directory for storage
        
    Returns:
        VectorStore instance
    """
    if store_type.lower() == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    elif store_type.lower() == "faiss":
        return FAISSVectorStore(
            index_path=persist_directory
        )
    else:
        raise ValueError(f"Unknown store_type: {store_type}. Use 'chroma' or 'faiss'")