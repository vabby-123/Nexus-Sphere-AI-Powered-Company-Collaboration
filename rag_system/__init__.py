"""
Nexus Sphere Advanced RAG System
Production-grade RAG with FREE embeddings (no API limits)
"""

from .data_fetchers import (
    SECFilingsFetcher, 
    PDFBatchProcessor, 
    URLScraper, 
    DocumentMetadata
)

from .document_processors import (
    AdvancedDocumentProcessor, 
    DocumentDeduplicator, 
    DocumentFilter
)

# Use free vector store (ChromaDB with HuggingFace embeddings)
from .vector_store_free import (
    ChromaVectorStore, 
    HybridSearchEngine,
    create_vector_store
)

from .advanced_rag import AdvancedRAGSystem

from .retrieval_strategies import (
    MultiQueryRetriever,
    ContextualCompressionRetriever,
    ReRankingRetriever
)

from .rag_ui import show_advanced_rag_system

__version__ = "1.0.0"

__all__ = [
    'SECFilingsFetcher',
    'PDFBatchProcessor',
    'URLScraper',
    'DocumentMetadata',
    'AdvancedDocumentProcessor',
    'DocumentDeduplicator',
    'DocumentFilter',
    'ChromaVectorStore',
    'HybridSearchEngine',
    'create_vector_store',
    'AdvancedRAGSystem',
    'MultiQueryRetriever',
    'ContextualCompressionRetriever',
    'ReRankingRetriever',
    'show_advanced_rag_system'
]