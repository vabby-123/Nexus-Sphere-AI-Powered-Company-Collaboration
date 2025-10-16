"""
Advanced retrieval strategies for RAG
"""

from typing import List, Dict, Any, Optional
import logging
from langchain_google_genai import GoogleGenerativeAI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """
    Generate multiple query variations for better retrieval
    
    Improves recall by querying from different angles
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.7
        )
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        num_queries: int = 3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple query variations and combine results
        
        Args:
            query: Original query
            filter_dict: Metadata filters
            num_queries: Number of query variations
            limit: Results per query
            
        Returns:
            Combined and deduplicated results
        """
        # Generate query variations
        variations = self._generate_query_variations(query, num_queries)
        
        logger.info(f"Generated {len(variations)} query variations")
        
        # Search with each variation
        all_results = []
        seen_ids = set()
        
        for variation in variations:
            results = self.vector_store.search(
                query=variation,
                filter_dict=filter_dict,
                limit=limit
            )
            
            # Deduplicate by ID
            for result in results:
                result_id = result.get('id')
                if result_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result_id)
        
        # Sort by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Multi-query retrieval: {len(all_results)} unique results")
        
        return all_results[:limit * 2]  # Return more for potential re-ranking
    
    def _generate_query_variations(self, query: str, num_variations: int) -> List[str]:
        """Generate query variations using LLM"""
        prompt = f"""Generate {num_variations} different ways to ask the following question. Each variation should capture different aspects or phrasings while maintaining the core intent.

Original question: {query}

Provide {num_variations} variations, one per line, without numbering or bullet points:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse variations
            variations = [query]  # Include original
            
            lines = response.strip().split('\n')
            for line in lines:
                cleaned = line.strip().lstrip('123456789.-) ')
                if cleaned and cleaned != query:
                    variations.append(cleaned)
            
            return variations[:num_variations + 1]
        
        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            return [query]


class ContextualCompressionRetriever:
    """
    Compress retrieved context to keep only relevant parts
    
    Reduces noise and improves answer quality
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.1
        )
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and compress documents
        
        Args:
            query: Search query
            filter_dict: Metadata filters
            limit: Number of results
            
        Returns:
            Compressed results
        """
        # Standard retrieval
        results = self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            limit=limit * 2  # Get more for compression
        )
        
        # Compress each result
        compressed_results = []
        
        for result in results[:limit]:
            compressed_text = self._compress_document(query, result['text'])
            
            if compressed_text:
                result['original_text'] = result['text']
                result['text'] = compressed_text
                result['compression_ratio'] = len(compressed_text) / len(result['text'])
                compressed_results.append(result)
        
        logger.info(
            f"Compressed {len(compressed_results)} documents "
            f"(avg compression: {sum(r['compression_ratio'] for r in compressed_results)/len(compressed_results):.1%})"
        )
        
        return compressed_results
    
    def _compress_document(self, query: str, document: str) -> str:
        """Compress document to relevant parts only"""
        # For very short documents, no compression needed
        if len(document) < 200:
            return document
        
        prompt = f"""Extract only the parts of the following text that are directly relevant to answering this question: "{query}"

Text:
{document}

Relevant parts (maintain original wording):"""
        
        try:
            compressed = self.llm.invoke(prompt)
            
            # If compression resulted in very little text, return original
            if len(compressed.strip()) < 50:
                return document
            
            return compressed.strip()
        
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return document


class ReRankingRetriever:
    """Re-rank search results using LLM"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Try to initialize Gemini, but don't fail if unavailable
        try:
            import os
            gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            
            if gemini_key:
                from langchain_google_genai import GoogleGenerativeAI
                self.llm = GoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=gemini_key,
                    temperature=0
                )
                logger.info("✅ Re-ranking LLM initialized")
            else:
                self.llm = None
                logger.warning("⚠️ No Gemini API key - re-ranking will use fallback scoring")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize re-ranking LLM: {e}")
            self.llm = None
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-rank results"""
        
        if not results:
            return []
        
        # If no LLM available, use fallback (just return top results by score)
        if not self.llm:
            logger.info("Using fallback re-ranking (no LLM)")
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            return sorted_results[:top_k]
        
        # Score each result
        scored_results = []
        
        for result in results:
            try:
                relevance_score = self._score_relevance(query, result['text'])
                
                # Combine with original similarity score
                combined_score = 0.6 * relevance_score + 0.4 * result.get('score', 0)
                
                result_with_score = result.copy()
                result_with_score['rerank_score'] = combined_score
                
                scored_results.append(result_with_score)
                
            except Exception as e:
                logger.error(f"Scoring error: {e}")
                # Keep original score if scoring fails
                result_with_score = result.copy()
                result_with_score['rerank_score'] = result.get('score', 0)
                scored_results.append(result_with_score)
        
        # Sort by re-ranking score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Re-ranked {len(results)} results, returning top {top_k}")
        
        return scored_results[:top_k]
    
    def _score_relevance(self, query: str, document: str) -> float:
        """Score document relevance to query"""
        # Truncate document if too long
        if len(document) > 2000:
            document = document[:2000] + "..."
        
        prompt = f"""Rate the relevance of this document to the query on a scale of 0.0 to 1.0, where:
- 0.0 = Completely irrelevant
- 0.5 = Somewhat relevant
- 1.0 = Highly relevant and directly answers the query

Query: {query}

Document:
{document}

Relevance score (just the number):"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Extract score
            score_str = response.strip().split()[0]
            score = float(score_str)
            
            # Clamp to valid range
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return 0.5  # Default middle score


class ParentDocumentRetriever:
    """
    Retrieve child chunks but return parent documents for context
    
    Useful when you want granular search but broader context
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search child documents and return parents
        
        Args:
            query: Search query
            filter_dict: Metadata filters
            limit: Number of parent documents
            
        Returns:
            Parent documents
        """
        # Search for child documents
        child_filter = filter_dict or {}
        child_filter['chunk_type'] = 'child'
        
        child_results = self.vector_store.search(
            query=query,
            filter_dict=child_filter,
            limit=limit * 3  # Get more children
        )
        
        # Get unique parent IDs
        parent_ids = set()
        for result in child_results:
            parent_id = result.get('metadata', {}).get('parent_id')
            if parent_id:
                parent_ids.add(parent_id)
        
        # Retrieve parent documents
        parent_results = []
        
        for parent_id in list(parent_ids)[:limit]:
            parent_filter = {'parent_id': parent_id, 'chunk_type': 'parent'}
            
            parents = self.vector_store.search(
                query=query,
                filter_dict=parent_filter,
                limit=1
            )
            
            if parents:
                parent_results.append(parents[0])
        
        logger.info(f"Retrieved {len(parent_results)} parent documents from {len(child_results)} children")
        
        return parent_results
