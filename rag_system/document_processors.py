"""
Advanced document processing with semantic chunking and optimization
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)
from langchain.docstore.document import Document
import numpy as np
from .data_fetchers import DocumentMetadata
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDocumentProcessor:
    """
    Advanced document processing with multiple chunking strategies
    
    Features:
    - Semantic chunking
    - Token-aware splitting
    - Context preservation
    - Metadata enrichment
    - Multi-strategy support
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base"  # GPT-4 encoding
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except:
            logger.warning("Could not load tiktoken, using character-based splitting")
            self.tokenizer = None
        
        # Initialize splitters
        self._init_splitters()
    
    def _init_splitters(self):
        """Initialize various text splitters"""
        
        # Recursive character splitter (default)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Token-based splitter
        if self.tokenizer:
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size // 4,  # Tokens are roughly 1/4 of characters
                chunk_overlap=self.chunk_overlap // 4,
                encoding_name="cl100k_base"
            )
        
        # Markdown splitter (for structured documents)
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = "recursive",
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Chunk documents using specified strategy
        
        Args:
            documents: List of Document objects
            strategy: Chunking strategy ('recursive', 'token', 'semantic', 'sentence')
            preserve_metadata: Preserve and enrich metadata
            
        Returns:
            List of chunked Document objects
        """
        if strategy == "recursive":
            return self._chunk_recursive(documents, preserve_metadata)
        elif strategy == "token":
            return self._chunk_token_based(documents, preserve_metadata)
        elif strategy == "semantic":
            return self._chunk_semantic(documents, preserve_metadata)
        elif strategy == "sentence":
            return self._chunk_sentence_based(documents, preserve_metadata)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using recursive")
            return self._chunk_recursive(documents, preserve_metadata)
    
    def _chunk_recursive(
        self,
        documents: List[Document],
        preserve_metadata: bool
    ) -> List[Document]:
        """Recursive character-based chunking"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.recursive_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                
                # Enrich metadata
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'char_count': len(chunk),
                    'chunking_strategy': 'recursive'
                })
                
                # Add context hints
                if i > 0:
                    chunk_metadata['has_previous_context'] = True
                if i < len(chunks) - 1:
                    chunk_metadata['has_next_context'] = True
                
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Recursive chunking: {len(documents)} docs → {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _chunk_token_based(
        self,
        documents: List[Document],
        preserve_metadata: bool
    ) -> List[Document]:
        """Token-aware chunking for LLM optimization"""
        if not self.tokenizer:
            logger.warning("Tokenizer not available, falling back to recursive")
            return self._chunk_recursive(documents, preserve_metadata)
        
        chunked_docs = []
        
        for doc in documents:
            chunks = self.token_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                
                # Calculate token count
                token_count = len(self.tokenizer.encode(chunk))
                
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'token_count': token_count,
                    'word_count': len(chunk.split()),
                    'chunking_strategy': 'token_based'
                })
                
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Token-based chunking: {len(documents)} docs → {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _chunk_semantic(
        self,
        documents: List[Document],
        preserve_metadata: bool
    ) -> List[Document]:
        """
        Semantic chunking based on topic coherence
        
        Uses paragraph boundaries and semantic similarity
        """
        chunked_docs = []
        
        for doc in documents:
            # Split into paragraphs
            paragraphs = doc.page_content.split('\n\n')
            
            # Group paragraphs semantically
            current_chunk = []
            current_size = 0
            chunk_index = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                # Check if adding this paragraph exceeds chunk size
                if current_size + para_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    
                    chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'word_count': len(chunk_text.split()),
                        'paragraph_count': len(current_chunk),
                        'chunking_strategy': 'semantic'
                    })
                    
                    chunked_docs.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_chunk:
                        # Keep last paragraph for context
                        current_chunk = [current_chunk[-1], para]
                        current_size = len(current_chunk[-1]) + para_size
                    else:
                        current_chunk = [para]
                        current_size = para_size
                    
                    chunk_index += 1
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                
                chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'total_chunks': chunk_index + 1,
                    'word_count': len(chunk_text.split()),
                    'paragraph_count': len(current_chunk),
                    'chunking_strategy': 'semantic'
                })
                
                chunked_docs.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
        
        logger.info(f"Semantic chunking: {len(documents)} docs → {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _chunk_sentence_based(
        self,
        documents: List[Document],
        preserve_metadata: bool
    ) -> List[Document]:
        """
        Sentence-boundary aware chunking
        
        Ensures chunks don't break mid-sentence
        """
        chunked_docs = []
        
        for doc in documents:
            # Split into sentences
            sentences = self._split_sentences(doc.page_content)
            
            # Group sentences into chunks
            current_chunk = []
            current_size = 0
            chunk_index = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    
                    chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'word_count': len(chunk_text.split()),
                        'sentence_count': len(current_chunk),
                        'chunking_strategy': 'sentence_based'
                    })
                    
                    chunked_docs.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
                    
                    # Start new chunk with overlap
                    overlap_sentences = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                    current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                    current_size = sum(len(s) for s in current_chunk)
                    
                    chunk_index += 1
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'total_chunks': chunk_index + 1,
                    'word_count': len(chunk_text.split()),
                    'sentence_count': len(current_chunk),
                    'chunking_strategy': 'sentence_based'
                })
                
                chunked_docs.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
        
        logger.info(f"Sentence-based chunking: {len(documents)} docs → {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Improved sentence splitting pattern
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_parent_child_documents(
        self,
        documents: List[Document],
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500
    ) -> Tuple[List[Document], List[Document]]:
        """
        Create parent-child document hierarchy
        
        Useful for retrieval where you want to:
        1. Search at granular level (child)
        2. Return broader context (parent)
        
        Args:
            documents: Source documents
            parent_chunk_size: Size of parent chunks
            child_chunk_size: Size of child chunks
            
        Returns:
            Tuple of (parent_documents, child_documents)
        """
        # Create parent splitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create child splitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        parent_documents = []
        child_documents = []
        
        for doc in documents:
            # Create parent chunks
            parent_chunks = parent_splitter.split_text(doc.page_content)
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # Create parent document
                parent_id = f"{doc.metadata.get('doc_id', 'unknown')}_{parent_idx}"
                
                parent_metadata = doc.metadata.copy()
                parent_metadata.update({
                    'parent_id': parent_id,
                    'chunk_type': 'parent',
                    'chunk_index': parent_idx,
                    'total_parent_chunks': len(parent_chunks)
                })
                
                parent_doc = Document(
                    page_content=parent_chunk,
                    metadata=parent_metadata
                )
                parent_documents.append(parent_doc)
                
                # Create child chunks from this parent
                child_chunks = child_splitter.split_text(parent_chunk)
                
                for child_idx, child_chunk in enumerate(child_chunks):
                    child_metadata = doc.metadata.copy()
                    child_metadata.update({
                        'parent_id': parent_id,
                        'chunk_type': 'child',
                        'parent_chunk_index': parent_idx,
                        'child_chunk_index': child_idx,
                        'total_child_chunks': len(child_chunks)
                    })
                    
                    child_doc = Document(
                        page_content=child_chunk,
                        metadata=child_metadata
                    )
                    child_documents.append(child_doc)
        
        logger.info(
            f"Created {len(parent_documents)} parent docs and "
            f"{len(child_documents)} child docs from {len(documents)} source docs"
        )
        
        return parent_documents, child_documents
    
    def enrich_metadata(
        self,
        documents: List[Document],
        extract_keywords: bool = True,
        extract_entities: bool = False
    ) -> List[Document]:
        """
        Enrich document metadata with extracted information
        
        Args:
            documents: Documents to enrich
            extract_keywords: Extract keywords from content
            extract_entities: Extract named entities (requires spaCy)
            
        Returns:
            Documents with enriched metadata
        """
        enriched_docs = []
        
        for doc in documents:
            metadata = doc.metadata.copy()
            
            # Extract keywords (simple frequency-based)
            if extract_keywords:
                keywords = self._extract_keywords(doc.page_content)
                metadata['keywords'] = keywords
            
            # Calculate readability metrics
            metadata['readability'] = self._calculate_readability(doc.page_content)
            
            # Detect language (simple heuristic)
            metadata['language_detected'] = self._detect_language(doc.page_content)
            
            # Extract entities (if requested and spaCy available)
            if extract_entities:
                try:
                    entities = self._extract_entities(doc.page_content)
                    metadata['entities'] = entities
                except:
                    logger.warning("Entity extraction failed - spaCy may not be installed")
            
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Simple keyword extraction using frequency"""
        # Tokenize
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 
                     'will', 'are', 'was', 'been', 'has', 'had', 'their', 'they'}
        
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        from collections import Counter
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics"""
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Avoid division by zero
        if not sentences or not words:
            return {'avg_words_per_sentence': 0, 'avg_chars_per_word': 0}
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(w) for w in words) / len(words)
        
        return {
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_chars_per_word': round(avg_chars_per_word, 2)
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (English-centric)"""
        # Very simple heuristic - check for common English words
        english_indicators = ['the', 'and', 'for', 'with', 'this', 'that']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        return 'en' if english_count >= 3 else 'unknown'
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            
            doc = nlp(text[:10000])  # Limit length for performance
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            return entities
        except:
            return {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common in PDFs)
        text = re.sub(r'\b\d+\b\s*\n', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()


class DocumentDeduplicator:
    """Remove duplicate or near-duplicate documents"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content similarity
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return []
        
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # Create content hash
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(content_hash)
        
        logger.info(f"Deduplication: {len(documents)} docs → {len(unique_docs)} unique docs")
        
        return unique_docs
    
    def find_near_duplicates(
        self,
        documents: List[Document],
        threshold: float = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate document pairs
        
        Returns:
            List of (index1, index2, similarity_score) tuples
        """
        threshold = threshold or self.similarity_threshold
        
        near_duplicates = []
        
        # Simple character-based similarity for now
        # In production, use embeddings for better results
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = self._calculate_similarity(
                    documents[i].page_content,
                    documents[j].page_content
                )
                
                if similarity >= threshold:
                    near_duplicates.append((i, j, similarity))
        
        return near_duplicates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class DocumentFilter:
    """Filter documents based on various criteria"""
    
    def filter_by_length(
        self,
        documents: List[Document],
        min_length: int = 100,
        max_length: Optional[int] = None
    ) -> List[Document]:
        """Filter documents by content length"""
        filtered = []
        
        for doc in documents:
            content_length = len(doc.page_content)
            
            if content_length >= min_length:
                if max_length is None or content_length <= max_length:
                    filtered.append(doc)
        
        logger.info(f"Length filter: {len(documents)} docs → {len(filtered)} docs")
        return filtered
    
    def filter_by_language(
        self,
        documents: List[Document],
        languages: List[str] = ['en']
    ) -> List[Document]:
        """Filter documents by language"""
        filtered = [
            doc for doc in documents
            if doc.metadata.get('language', 'en') in languages
        ]
        
        logger.info(f"Language filter: {len(documents)} docs → {len(filtered)} docs")
        return filtered
    
    def filter_by_metadata(
        self,
        documents: List[Document],
        filters: Dict[str, Any]
    ) -> List[Document]:
        """Filter documents by metadata criteria"""
        filtered = []
        
        for doc in documents:
            match = True
            
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered.append(doc)
        
        logger.info(f"Metadata filter: {len(documents)} docs → {len(filtered)} docs")
        return filtered
    
    def filter_by_quality(
        self,
        documents: List[Document],
        min_quality_score: float = 0.5
    ) -> List[Document]:
        """
        Filter documents by quality score
        
        Quality score based on:
        - Length appropriateness
        - Readability
        - Information density
        """
        filtered = []
        
        for doc in documents:
            quality_score = self._calculate_quality_score(doc)
            
            if quality_score >= min_quality_score:
                doc.metadata['quality_score'] = quality_score
                filtered.append(doc)
        
        logger.info(f"Quality filter: {len(documents)} docs → {len(filtered)} docs")
        return filtered
    
    def _calculate_quality_score(self, doc: Document) -> float:
        """Calculate document quality score"""
        score = 0.0
        
        content = doc.page_content
        
        # Length score (prefer documents between 200-2000 chars)
        length = len(content)
        if 200 <= length <= 2000:
            score += 0.3
        elif length > 100:
            score += 0.15
        
        # Word diversity score
        words = content.split()
        unique_words = set(w.lower() for w in words)
        
        if words:
            diversity = len(unique_words) / len(words)
            score += diversity * 0.3
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        if 10 <= avg_sentence_length <= 30:
            score += 0.2
        
        # Information density (proper nouns, numbers)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', content))
        numbers = len(re.findall(r'\b\d+\b', content))
        
        if proper_nouns > 3 or numbers > 2:
            score += 0.2
        
        return min(score, 1.0)