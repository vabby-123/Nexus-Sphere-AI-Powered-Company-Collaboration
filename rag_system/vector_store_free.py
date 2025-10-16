"""
Production-grade vector store with ChromaDB (no Docker) - FIXED VERSION
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging
from pathlib import Path

import numpy as np
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store with proper persistence
    """
    
    def __init__(
        self,
        collection_name: str = "nexus_sphere_knowledge",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB vector store"""
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model")
        
        # Initialize ChromaDB client (NEW API)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB at {self.persist_directory}")
        logger.info(f" Current document count: {self.collection.count()}")
    
    async def add_documents_batch(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """Add documents in batches with proper persistence"""
        if not documents:
            return 0
        
        total_added = 0
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Adding {len(documents)} documents in {num_batches} batches")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract data
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                # Generate embeddings
                embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents,
                    texts
                )
                
                # Generate unique IDs
                ids = [
                    hashlib.md5(
                        f"{doc.metadata.get('doc_id', '')}_{doc.page_content[:100]}".encode()
                    ).hexdigest()
                    for doc in batch
                ]
                
                # Add to ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                total_added += len(batch)
                logger.info(f"Batch {i//batch_size + 1}/{num_batches}: Added {len(batch)} documents")
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f" Successfully added {total_added} documents")
        logger.info(f" Total documents in collection: {self.collection.count()}")
        
        return total_added
    
    def search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build where clause for filtering
            where = self._build_where_clause(filter_dict) if filter_dict else None
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity score
                    distance = results['distances'][0][i]
                    similarity_score = 1 / (1 + distance)
                    
                    # Apply score threshold
                    if score_threshold and similarity_score < score_threshold:
                        continue
                    
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': similarity_score
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_where_clause(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filter dictionary"""
        where = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, list):
                # Multiple values (OR condition)
                where[key] = {"$in": value}
            elif isinstance(value, dict):
                # Range query
                if 'gte' in value:
                    where[key] = {"$gte": value['gte']}
                elif 'lte' in value:
                    where[key] = {"$lte": value['lte']}
            else:
                # Single value (exact match)
                where[key] = {"$eq": value}
        
        return where
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents matching filter"""
        try:
            where = self._build_where_clause(filter_dict)
            
            # Get matching documents
            results = self.collection.get(where=where)
            
            if results['ids']:
                # Delete by IDs
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents")
                return len(results['ids'])
            
            return 0
        
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            return {
                'name': self.collection_name,
                'points_count': count,
                'persist_directory': str(self.persist_directory),
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'status': 'active' if count > 0 else 'empty',
                'vector_size': 384,  # all-MiniLM-L6-v2 embedding size
                'distance': 'cosine'
            }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'name': self.collection_name,
                'points_count': 0,
                'status': 'error'
            }
    
    def scroll_points(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict], Optional[int]]:
        """Scroll through collection points"""
        try:
            where = self._build_where_clause(filter_dict) if filter_dict else None
            
            results = self.collection.get(
                where=where,
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


class HybridSearchEngine:
    """Hybrid search combining vector and keyword search"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def hybrid_search(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword methods"""
        # For now, just use vector search
        # Full hybrid implementation would require BM25 index
        return self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            limit=limit
        )


def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "nexus_sphere_knowledge",
    persist_directory: str = "./chroma_db"
):
    """Factory function to create vector store"""
    if store_type.lower() == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    else:
        raise ValueError(f"Unknown store_type: {store_type}")