# """
# Main RAG orchestrator - integrates all components
# """

# import asyncio
# from typing import List, Dict, Any, Optional, Callable
# from pathlib import Path
# import logging
# from datetime import datetime
# import json
# import os  # ADD THIS!
# import numpy as np

# from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# # FIX: Add dots for relative imports
# from .data_fetchers import (
#     SECFilingsFetcher,
#     PDFBatchProcessor,
#     URLScraper,
#     DocumentMetadata
# )
# from .document_processors import (
#     AdvancedDocumentProcessor,
#     DocumentDeduplicator,
#     DocumentFilter
# )
# from .vector_store_free import ChromaVectorStore, create_vector_store, HybridSearchEngine
# from .retrieval_strategies import (
#     MultiQueryRetriever,
#     ContextualCompressionRetriever,
#     ReRankingRetriever
# )

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class AdvancedRAGSystem:
#     """
#     Production-grade RAG system orchestrator
    
#     Features:
#     - Multi-source data ingestion (SEC, PDFs, URLs)
#     - Advanced document processing
#     - Hybrid search
#     - Multiple retrieval strategies
#     - Query optimization
#     - Result re-ranking
#     """
    
#     def __init__(
#         self,
#         db: 'UnifiedDatabase',
#         collection_name: str = "nexus_sphere_knowledge",
#         vector_store_type: str = "chroma",  # "chroma" or "faiss"
#         persist_directory: str = "./chroma_db"
#     ):
#         """
#         Initialize Advanced RAG System
        
#         Args:
#             db: UnifiedDatabase instance
#             collection_name: Collection name for vector store
#             vector_store_type: "chroma" or "faiss"
#             persist_directory: Directory for vector store persistence
#         """
#         self.db = db
        
#         # Initialize data fetchers
#         # Option 1: Use sec-api.io (premium, recommended)
# # Initialize SEC fetcher
# # Initialize SEC fetcher
#         sec_api_key = os.getenv('SEC_API_KEY')
#         if sec_api_key:
#             try:
#                 from .sec_api_fetcher import SecApiFetcher
#                 self.sec_fetcher = SecApiFetcher(api_key=sec_api_key)
#                 logger.info("✅ Using sec-api.io for SEC filings")
#             except Exception as e:
#                 logger.error(f"Failed to load sec-api.io: {e}")
#                 logger.info("⚠️ Falling back to free EDGAR API")
#                 self.sec_fetcher = SECFilingsFetcher(
#                     email=os.getenv('SEC_USER_EMAIL', 'user@example.com')
#                 )
#         else:
#             logger.info("⚠️ No SEC_API_KEY found - using free EDGAR API")
#             self.sec_fetcher = SECFilingsFetcher(
#                 email=os.getenv('SEC_USER_EMAIL', 'user@example.com')
#             )
                    
#         self.pdf_processor = PDFBatchProcessor()
#         self.url_scraper = URLScraper()
        
#         # Initialize document processors
#         self.doc_processor = AdvancedDocumentProcessor()
#         self.doc_deduplicator = DocumentDeduplicator()
#         self.doc_filter = DocumentFilter()
        
#         # Initialize vector store
#         self.vector_store = create_vector_store(
#             store_type=vector_store_type,
#             collection_name=collection_name,
#             persist_directory=persist_directory
#         )
#         self.hybrid_search = HybridSearchEngine(self.vector_store)
        
#         # Initialize retrieval strategies
#         from .retrieval_strategies import (
#             MultiQueryRetriever,
#             ContextualCompressionRetriever,
#             ReRankingRetriever
#         )
        
#         self.multi_query_retriever = MultiQueryRetriever(self.vector_store)
#         self.compression_retriever = ContextualCompressionRetriever(self.vector_store)
#         self.reranking_retriever = ReRankingRetriever(self.vector_store)
        
#         # Initialize LLM (optional - only if API key provided)
#         gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
#         if gemini_key:
#             try:
#                 from langchain_google_genai import GoogleGenerativeAI
#                 self.llm = GoogleGenerativeAI(
#                     model="gemini-2.0-flash-exp",
#                     google_api_key=gemini_key,
#                     temperature=0.2
#                 )
#                 logger.info("✅ Gemini LLM initialized")
#             except Exception as e:
#                 logger.warning(f"Could not initialize Gemini LLM: {e}")
#                 self.llm = None
#         else:
#             logger.info("ℹ️ No Gemini API key - LLM features will be limited")
#             self.llm = None
        
#         # Ingestion statistics
#         self.ingestion_stats = {
#             'total_documents': 0,
#             'sec_filings': 0,
#             'pdfs': 0,
#             'urls': 0,
#             'last_ingestion': None
#         }
        
#         logger.info("✅ Advanced RAG System initialized")
#         logger.info(f"   Vector Store: {vector_store_type}")
#         logger.info(f"   SEC Fetcher: {'sec-api.io' if sec_api_key else 'Free EDGAR'}")
#         logger.info(f"   LLM: {'Gemini' if self.llm else 'None'}")
    
#     # ========================================================================
#     # DATA INGESTION METHODS
#     # ========================================================================
    
#     async def ingest_sec_filings_for_companies(
#         self,
#         tickers: List[str],
#         filing_types: List[str] = ['10-K', '10-Q'],
#         count_per_type: int = 5,
#         progress_callback: Optional[Callable] = None
#     ) -> Dict[str, Any]:
#         """
#         Ingest SEC filings for multiple companies
        
#         Args:
#             tickers: List of stock tickers
#             filing_types: Types of filings to fetch
#             count_per_type: Number of each filing type
#             progress_callback: Callback function(ticker, progress, status)
            
#         Returns:
#             Ingestion statistics
#         """
#         logger.info(f"Starting SEC ingestion for {len(tickers)} companies")
        
#         stats = {
#             'tickers_processed': 0,
#             'filings_downloaded': 0,
#             'documents_created': 0,
#             'chunks_indexed': 0,
#             'errors': []
#         }
        
#         for i, ticker in enumerate(tickers):
#             try:
#                 if progress_callback:
#                     progress_callback(ticker, i / len(tickers), f"Processing {ticker}")
                
#                 # Fetch filings metadata
#                 filings = await self.sec_fetcher.fetch_company_filings(
#                     ticker=ticker,
#                     filing_types=filing_types,
#                     count=count_per_type
#                 )
                
#                 logger.info(f"{ticker}: Found {len(filings)} filings")
                
#                 # Process each filing
#                 for filing in filings:
#                     # Download filing
#                     file_path = await self.sec_fetcher.download_filing(filing)
                    
#                     if not file_path:
#                         continue
                    
#                     stats['filings_downloaded'] += 1
                    
#                     # Extract text and sections
#                     sections = self.sec_fetcher.extract_text_from_filing(file_path)
                    
#                     # Create documents
#                     documents = []
                    
#                     for section_name, section_text in sections.items():
#                         # Create metadata
#                         metadata = DocumentMetadata(
#                             doc_id=f"{ticker}_{filing['filing_type']}_{filing['filing_date']}_{section_name}",
#                             source_type='sec_filing',
#                             company_name=ticker,
#                             ticker=ticker,
#                             filing_type=filing['filing_type'],
#                             fiscal_year=int(filing['filing_date'][:4]),
#                             url=filing['url'],
#                             file_path=str(file_path),
#                             document_date=filing['filing_date'],
#                             section=section_name
#                         )
                        
#                         # Clean text
#                         cleaned_text = self.doc_processor.clean_text(section_text)
                        
#                         doc = Document(
#                             page_content=cleaned_text,
#                             metadata=metadata.to_dict()
#                         )
                        
#                         documents.append(doc)
                    
#                     stats['documents_created'] += len(documents)
                    
#                     # Chunk documents
#                     chunked_docs = self.doc_processor.chunk_documents(
#                         documents,
#                         strategy='semantic'
#                     )
                    
#                     # Filter low-quality chunks
#                     filtered_docs = self.doc_filter.filter_by_length(
#                         chunked_docs,
#                         min_length=100
#                     )
                    
#                     # Add to vector store
#                     added = await self.vector_store.add_documents_batch(filtered_docs)
#                     stats['chunks_indexed'] += added
                    
#                     logger.info(
#                         f"{ticker} {filing['filing_type']} {filing['filing_date']}: "
#                         f"Indexed {added} chunks"
#                     )
                
#                 stats['tickers_processed'] += 1
                
#             except Exception as e:
#                 logger.error(f"Error processing {ticker}: {e}")
#                 stats['errors'].append({'ticker': ticker, 'error': str(e)})
        
#         # Update global stats
#         self.ingestion_stats['sec_filings'] += stats['chunks_indexed']
#         self.ingestion_stats['total_documents'] += stats['chunks_indexed']
#         self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
#         if progress_callback:
#             progress_callback(None, 1.0, "Ingestion complete")
        
#         logger.info(f"SEC ingestion complete: {stats}")
        
#         return stats
    
#     async def ingest_pdfs_from_directory(
#         self,
#         directory: Path,
#         source_type: str = 'financial_report',
#         recursive: bool = True,
#         metadata_extractor: Optional[Callable] = None,
#         progress_callback: Optional[Callable] = None
#     ) -> Dict[str, Any]:
#         """
#         Ingest all PDFs from a directory
        
#         Args:
#             directory: Directory path
#             source_type: Document source type
#             recursive: Search subdirectories
#             metadata_extractor: Function to extract metadata from filename
#             progress_callback: Progress callback
            
#         Returns:
#             Ingestion statistics
#         """
#         logger.info(f"Starting PDF ingestion from {directory}")
        
#         stats = {
#             'pdfs_found': 0,
#             'pdfs_processed': 0,
#             'documents_created': 0,
#             'chunks_indexed': 0,
#             'errors': []
#         }
        
#         # Find all PDFs
#         if recursive:
#             pdf_files = list(directory.rglob('*.pdf'))
#         else:
#             pdf_files = list(directory.glob('*.pdf'))
        
#         stats['pdfs_found'] = len(pdf_files)
#         logger.info(f"Found {len(pdf_files)} PDF files")
        
#         for i, pdf_path in enumerate(pdf_files):
#             try:
#                 if progress_callback:
#                     progress_callback(
#                         pdf_path.name,
#                         i / len(pdf_files),
#                         f"Processing {pdf_path.name}"
#                     )
                
#                 # Extract metadata from filename
#                 if metadata_extractor:
#                     file_metadata = metadata_extractor(pdf_path.name)
#                 else:
#                     file_metadata = self.pdf_processor.extract_metadata_from_filename(
#                         pdf_path.name
#                     )
                
#                 # Process PDF
#                 pdf_data = self.pdf_processor.process_pdf(pdf_path)
                
#                 if not pdf_data['success']:
#                     stats['errors'].append({
#                         'file': pdf_path.name,
#                         'error': pdf_data.get('error', 'Unknown error')
#                     })
#                     continue
                
#                 stats['pdfs_processed'] += 1
                
#                 # Create documents from pages
#                 documents = []
                
#                 for page_data in pdf_data['pages']:
#                     metadata = DocumentMetadata(
#                         doc_id=f"{pdf_path.stem}_page_{page_data['page_number']}",
#                         source_type=source_type,
#                         file_path=str(pdf_path),
#                         page_number=page_data['page_number'],
#                         **file_metadata
#                     )
                    
#                     # Clean text
#                     cleaned_text = self.doc_processor.clean_text(page_data['text'])
                    
#                     doc = Document(
#                         page_content=cleaned_text,
#                         metadata=metadata.to_dict()
#                     )
                    
#                     documents.append(doc)
                
#                 stats['documents_created'] += len(documents)
                
#                 # Chunk documents
#                 chunked_docs = self.doc_processor.chunk_documents(
#                     documents,
#                     strategy='recursive'
#                 )
                
#                 # Filter and deduplicate
#                 filtered_docs = self.doc_filter.filter_by_length(
#                     chunked_docs,
#                     min_length=100
#                 )
                
#                 unique_docs = self.doc_deduplicator.deduplicate(filtered_docs)
                
#                 # Add to vector store
#                 added = await self.vector_store.add_documents_batch(unique_docs)
#                 stats['chunks_indexed'] += added
                
#                 logger.info(f"{pdf_path.name}: Indexed {added} chunks")
                
#             except Exception as e:
#                 logger.error(f"Error processing {pdf_path}: {e}")
#                 stats['errors'].append({'file': pdf_path.name, 'error': str(e)})
        
#         # Update global stats
#         self.ingestion_stats['pdfs'] += stats['chunks_indexed']
#         self.ingestion_stats['total_documents'] += stats['chunks_indexed']
#         self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
#         if progress_callback:
#             progress_callback(None, 1.0, "PDF ingestion complete")
        
#         logger.info(f"PDF ingestion complete: {stats}")
        
#         return stats
    
#     async def ingest_urls(
#         self,
#         urls: List[Dict[str, Any]],
#         progress_callback: Optional[Callable] = None
#     ) -> Dict[str, Any]:
#         """
#         Ingest content from URLs
        
#         Args:
#             urls: List of URL dictionaries with metadata
#             progress_callback: Progress callback
            
#         Returns:
#             Ingestion statistics
#         """
#         logger.info(f"Starting URL ingestion for {len(urls)} URLs")
        
#         stats = {
#             'urls_processed': 0,
#             'documents_created': 0,
#             'chunks_indexed': 0,
#             'errors': []
#         }
        
#         for i, url_data in enumerate(urls):
#             try:
#                 url = url_data['url']
                
#                 if progress_callback:
#                     progress_callback(url, i / len(urls), f"Scraping {url}")
                
#                 # Scrape URL
#                 scraped_data = await self.url_scraper.scrape_url(url)
                
#                 if scraped_data['status'] != 'success':
#                     stats['errors'].append({
#                         'url': url,
#                         'error': scraped_data.get('error', 'Unknown error')
#                     })
#                     continue
                
#                 stats['urls_processed'] += 1
                
#                 # Create metadata
#                 metadata = DocumentMetadata(
#                     doc_id=scraped_data['metadata'].get('url'),
#                     source_type=url_data.get('source_type', 'web_article'),
#                     url=url,
#                     company_name=url_data.get('company_name'),
#                     ticker=url_data.get('ticker'),
#                     document_date=scraped_data['scraped_at']
#                 )
                
#                 # Update metadata with scraped info
#                 metadata_dict = metadata.to_dict()
#                 metadata_dict.update(scraped_data['metadata'])
                
#                 # Clean text
#                 cleaned_text = self.doc_processor.clean_text(scraped_data['content'])
                
#                 doc = Document(
#                     page_content=cleaned_text,
#                     metadata=metadata_dict
#                 )
                
#                 stats['documents_created'] += 1
                
#                 # Chunk document
#                 chunked_docs = self.doc_processor.chunk_documents(
#                     [doc],
#                     strategy='recursive'
#                 )
                
#                 # Filter
#                 filtered_docs = self.doc_filter.filter_by_length(
#                     chunked_docs,
#                     min_length=100
#                 )
                
#                 # Add to vector store
#                 added = await self.vector_store.add_documents_batch(filtered_docs)
#                 stats['chunks_indexed'] += added
                
#                 logger.info(f"{url}: Indexed {added} chunks")
                
#             except Exception as e:
#                 logger.error(f"Error processing {url_data.get('url')}: {e}")
#                 stats['errors'].append({
#                     'url': url_data.get('url'),
#                     'error': str(e)
#                 })
        
#         # Update global stats
#         self.ingestion_stats['urls'] += stats['chunks_indexed']
#         self.ingestion_stats['total_documents'] += stats['chunks_indexed']
#         self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
#         if progress_callback:
#             progress_callback(None, 1.0, "URL ingestion complete")
        
#         logger.info(f"URL ingestion complete: {stats}")
        
#         return stats
    
#     # ========================================================================
#     # QUERY METHODS
#     # ========================================================================
    
#     def query(
#         self,
#         question: str,
#         retrieval_strategy: str = "hybrid",
#         filter_dict: Optional[Dict[str, Any]] = None,
#         num_sources: int = 5,
#         use_reranking: bool = True
#     ) -> Dict[str, Any]:
#         """
#         Query the RAG system
        
#         Args:
#             question: User question
#             retrieval_strategy: 'hybrid', 'vector', 'multi_query', 'compression'
#             filter_dict: Metadata filters
#             num_sources: Number of sources to retrieve
#             use_reranking: Apply re-ranking to results
            
#         Returns:
#             Answer with sources and metadata
#         """
#         try:
#             # Retrieve documents based on strategy
#             if retrieval_strategy == "hybrid":
#                 results = self.hybrid_search.hybrid_search(
#                     query=question,
#                     filter_dict=filter_dict,
#                     limit=num_sources * 2 if use_reranking else num_sources
#                 )
#             elif retrieval_strategy == "multi_query":
#                 results = self.multi_query_retriever.retrieve(
#                     query=question,
#                     filter_dict=filter_dict,
#                     num_queries=3,
#                     limit=num_sources * 2 if use_reranking else num_sources
#                 )
#             elif retrieval_strategy == "compression":
#                 results = self.compression_retriever.retrieve(
#                     query=question,
#                     filter_dict=filter_dict,
#                     limit=num_sources * 2 if use_reranking else num_sources
#                 )
#             else:  # Default to vector search
#                 results = self.vector_store.search(
#                     query=question,
#                     filter_dict=filter_dict,
#                     limit=num_sources * 2 if use_reranking else num_sources
#                 )
            
#             if not results:
#                 return {
#                     'answer': 'No relevant information found in the knowledge base.',
#                     'sources': [],
#                     'confidence': 0.0,
#                     'retrieval_strategy': retrieval_strategy,
#                     'num_sources_used': 0
#                 }
            
#             # Apply re-ranking if requested
#             if use_reranking:
#                 results = self.reranking_retriever.rerank(
#                     query=question,
#                     results=results,
#                     top_k=num_sources
#                 )
            
#             # Build context
#             context = self._build_context(results)
            
#             # Generate answer
#             answer = self._generate_answer(question, context)
            
#             # Calculate confidence
#             confidence = self._calculate_confidence(results)
            
#             return {
#                 'answer': answer,
#                 'sources': self._format_sources(results),
#                 'confidence': confidence,
#                 'retrieval_strategy': retrieval_strategy,
#                 'num_sources_used': len(results)
#             }
        
#         except Exception as e:
#             logger.error(f"Query failed: {e}")
#             return {
#                 'error': str(e),
#                 'answer': 'An error occurred while processing your query.',
#                 'sources': [],
#                 'num_sources_used': 0,  # <-- Add this
#                 'confidence': 0.0,  # <-- Add this
#                 'retrieval_strategy': retrieval_strategy  # <-- Add this
#             }
    
#     def _build_context(self, results: List[Dict[str, Any]]) -> str:
#         """Build context from search results"""
#         context_parts = []
        
#         for i, result in enumerate(results, 1):
#             source_info = f"Source {i}"
            
#             # Add metadata info
#             metadata = result.get('metadata', {})
#             if metadata.get('ticker'):
#                 source_info += f" ({metadata['ticker']}"
#                 if metadata.get('filing_type'):
#                     source_info += f", {metadata['filing_type']}"
#                 if metadata.get('fiscal_year'):
#                     source_info += f", {metadata['fiscal_year']}"
#                 source_info += ")"
            
#             context_parts.append(f"{source_info}:\n{result['text']}\n")
        
#         return "\n---\n".join(context_parts)
    
#     def _generate_answer(self, question: str, context: str) -> str:
#         """Generate answer using LLM"""
#         prompt = f"""Based on the following information from financial documents, answer the question accurately and comprehensively.

# Context:
# {context}

# Question: {question}

# Instructions:
# - Provide a detailed, accurate answer based solely on the context provided
# - Cite specific sources when making claims (e.g., "According to Source 1...")
# - If the context doesn't contain enough information, acknowledge this
# - Be objective and factual
# - Include relevant numbers, dates, and specifics when available
# - If comparing companies or time periods, be explicit about the differences

# Answer:"""
        
#         try:
#             answer = self.llm.invoke(prompt)
#             return answer
#         except Exception as e:
#             logger.error(f"LLM generation failed: {e}")
#             return "I apologize, but I encountered an error generating the answer."
    
#     def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
#         """Calculate confidence score based on retrieval quality"""
#         if not results:
#             return 0.0
        
#         # Get scores
#         scores = [r.get('score', 0) for r in results]
        
#         # Calculate metrics
#         avg_score = sum(scores) / len(scores)
#         max_score = max(scores)
#         score_variance = np.var(scores) if len(scores) > 1 else 0
        
#         # Confidence calculation
#         # - High average score = good
#         # - High max score = at least one very relevant result
#         # - Low variance = consistent relevance
        
#         confidence = (
#             0.5 * avg_score +
#             0.3 * max_score +
#             0.2 * (1 - min(score_variance, 1))
#         )
        
#         return min(confidence, 1.0)
    
#     def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Format sources for display"""
#         formatted = []
        
#         for result in results:
#             metadata = result.get('metadata', {})
            
#             source = {
#                 'text': result['text'][:500] + '...',
#                 'relevance_score': result.get('score', 0),
#                 'source_type': metadata.get('source_type', 'unknown'),
#                 'ticker': metadata.get('ticker'),
#                 'company_name': metadata.get('company_name'),
#                 'filing_type': metadata.get('filing_type'),
#                 'fiscal_year': metadata.get('fiscal_year'),
#                 'section': metadata.get('section'),
#                 'url': metadata.get('url'),
#                 'document_date': metadata.get('document_date')
#             }
            
#             # Remove None values
#             source = {k: v for k, v in source.items() if v is not None}
            
#             formatted.append(source)
        
#         return formatted
    
#     # ========================================================================
#     # UTILITY METHODS
#     # ========================================================================
    
#     def get_statistics(self) -> Dict[str, Any]:
#         """Get system statistics"""
#         collection_info = self.vector_store.get_collection_info()
        
#         return {
#             'ingestion_stats': self.ingestion_stats,
#             'collection_info': collection_info,
#             'retrieval_strategies': [
#                 'hybrid',
#                 'vector',
#                 'multi_query',
#                 'compression'
#             ],
#             'supported_sources': ['sec_filings', 'pdfs', 'urls']
#         }
    
#     def find_similar_partnerships(
#         self,
#         company1_id: int,
#         company2_id: int,
#         limit: int = 5
#     ) -> List[Dict[str, Any]]:
#         """
#         Find similar historical partnerships
        
#         Args:
#             company1_id: First company ID
#             company2_id: Second company ID
#             limit: Number of results
            
#         Returns:
#             List of similar partnership analyses
#         """
#         # Get company profiles
#         company1 = self.db.get_company_profile(company1_id)
#         company2 = self.db.get_company_profile(company2_id)
        
#         if not company1 or not company2:
#             return []
        
#         # Build query
#         query = f"""
#         Find partnerships and analyses similar to:
#         Company A: {company1.name} in {company1.industry} (Revenue: ${company1.revenue/1e9:.1f}B)
#         Company B: {company2.name} in {company2.industry} (Revenue: ${company2.revenue/1e9:.1f}B)
        
#         Focus on partnerships with similar:
#         - Industry combinations
#         - Company sizes
#         - Revenue ranges
#         - Strategic objectives
#         """
        
#         # Search with filters
#         filter_dict = {
#             'source_type': ['sec_filing', 'research_report', 'case_study']
#         }
        
#         results = self.vector_store.search(
#             query=query,
#             filter_dict=filter_dict,
#             limit=limit
#         )
        
#         return results
    
#     def delete_documents_by_filter(self, filter_dict: Dict[str, Any]) -> int:
#         """
#         Delete documents matching filter criteria
        
#         Useful for removing outdated data
#         """
#         return self.vector_store.delete_by_filter(filter_dict)
    
#     def export_knowledge_base(
#         self,
#         output_file: Path,
#         filter_dict: Optional[Dict[str, Any]] = None
#     ):
#         """
#         Export knowledge base to JSON
        
#         Useful for backup or analysis
#         """
#         all_points = []
#         next_offset = None
        
#         while True:
#             points, next_offset = self.vector_store.scroll_points(
#                 filter_dict=filter_dict,
#                 limit=100,
#                 offset=next_offset
#             )
            
#             all_points.extend(points)
            
#             if next_offset is None:
#                 break
        
#         # Save to JSON
#         with open(output_file, 'w') as f:
#             json.dump(all_points, f, indent=2)
        
#         logger.info(f"Exported {len(all_points)} documents to {output_file}")
"""
Main RAG orchestrator - integrates all components
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import logging
from datetime import datetime
import json
import os
import numpy as np

from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# FIX: Add dots for relative imports
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
from .vector_store_free import ChromaVectorStore, create_vector_store, HybridSearchEngine




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HARDCODED API KEYS - WARNING: DO NOT COMMIT TO VERSION CONTROL!
# ============================================================================
HARDCODED_SEC_API_KEY = "your_sec_api_key_here"  # Replace with your actual key
HARDCODED_GEMINI_API_KEY = "AIzaSyByvZfBsfw0ZDhTeKXh5hL4GfBogIfac9c"
HARDCODED_SEC_USER_EMAIL = "your_email@example.com"  # Replace with your email
# ============================================================================

class AdvancedRAGSystem:
    """
    Production-grade RAG system orchestrator
    
    Features:
    - Multi-source data ingestion (SEC, PDFs, URLs)
    - Advanced document processing
    - Hybrid search
    - Multiple retrieval strategies
    - Query optimization
    - Result re-ranking
    """
    
    def __init__(
        self,
        db: 'UnifiedDatabase',
        collection_name: str = "nexus_sphere_knowledge",
        vector_store_type: str = "chroma",  # "chroma" or "faiss"
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize Advanced RAG System
        
        Args:
            db: UnifiedDatabase instance
            collection_name: Collection name for vector store
            vector_store_type: "chroma" or "faiss"
            persist_directory: Directory for vector store persistence
        """
        self.db = db
        
        # Initialize SEC fetcher with hardcoded key (fallback to env var)
        sec_api_key = HARDCODED_SEC_API_KEY if HARDCODED_SEC_API_KEY != "your_sec_api_key_here" else os.getenv('SEC_API_KEY')
        
        if sec_api_key:
            try:
                from .sec_api_fetcher import SecApiFetcher
                self.sec_fetcher = SecApiFetcher(api_key=sec_api_key)
                logger.info("✅ Using sec-api.io for SEC filings")
            except Exception as e:
                logger.error(f"Failed to load sec-api.io: {e}")
                logger.info("⚠️ Falling back to free EDGAR API")
                user_email = HARDCODED_SEC_USER_EMAIL if HARDCODED_SEC_USER_EMAIL != "your_email@example.com" else os.getenv('SEC_USER_EMAIL', 'user@example.com')
                self.sec_fetcher = SECFilingsFetcher(email=user_email)
        else:
            logger.info("⚠️ No SEC_API_KEY found - using free EDGAR API")
            user_email = HARDCODED_SEC_USER_EMAIL if HARDCODED_SEC_USER_EMAIL != "your_email@example.com" else os.getenv('SEC_USER_EMAIL', 'user@example.com')
            self.sec_fetcher = SECFilingsFetcher(email=user_email)
                    
        self.pdf_processor = PDFBatchProcessor()
        self.url_scraper = URLScraper()
        
        # Initialize document processors
        self.doc_processor = AdvancedDocumentProcessor()
        self.doc_deduplicator = DocumentDeduplicator()
        self.doc_filter = DocumentFilter()
        
        # Initialize vector store
        self.vector_store = create_vector_store(
            store_type=vector_store_type,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.hybrid_search = HybridSearchEngine(self.vector_store)
        
        # Initialize retrieval strategies
        from .retrieval_strategies import (
            MultiQueryRetriever,
            ContextualCompressionRetriever,
            ReRankingRetriever
        )
        
        self.multi_query_retriever = MultiQueryRetriever(self.vector_store)
        self.compression_retriever = ContextualCompressionRetriever(self.vector_store)
        self.reranking_retriever = ReRankingRetriever(self.vector_store)
        
        # Initialize LLM with hardcoded key (fallback to env var)
        gemini_key = HARDCODED_GEMINI_API_KEY if HARDCODED_GEMINI_API_KEY != "your_gemini_api_key_here" else (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
        
        if gemini_key:
            try:
                from langchain_google_genai import GoogleGenerativeAI
                self.llm = GoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=gemini_key,
                    temperature=0.2
                )
                logger.info("✅ Gemini LLM initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini LLM: {e}")
                self.llm = None
        else:
            logger.info("ℹ️ No Gemini API key - LLM features will be limited")
            self.llm = None
        
        # Ingestion statistics
        self.ingestion_stats = {
            'total_documents': 0,
            'sec_filings': 0,
            'pdfs': 0,
            'urls': 0,
            'last_ingestion': None
        }
        
        logger.info("✅ Advanced RAG System initialized")
        logger.info(f"   Vector Store: {vector_store_type}")
        logger.info(f"   SEC Fetcher: {'sec-api.io' if sec_api_key else 'Free EDGAR'}")
        logger.info(f"   LLM: {'Gemini' if self.llm else 'None'}")
    
    # ========================================================================
    # DATA INGESTION METHODS
    # ========================================================================
    
    async def ingest_sec_filings_for_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = ['10-K', '10-Q'],
        count_per_type: int = 5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest SEC filings for multiple companies
        
        Args:
            tickers: List of stock tickers
            filing_types: Types of filings to fetch
            count_per_type: Number of each filing type
            progress_callback: Callback function(ticker, progress, status)
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting SEC ingestion for {len(tickers)} companies")
        
        stats = {
            'tickers_processed': 0,
            'filings_downloaded': 0,
            'documents_created': 0,
            'chunks_indexed': 0,
            'errors': []
        }
        
        for i, ticker in enumerate(tickers):
            try:
                if progress_callback:
                    progress_callback(ticker, i / len(tickers), f"Processing {ticker}")
                
                # Fetch filings metadata
                filings = await self.sec_fetcher.fetch_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    count=count_per_type
                )
                
                logger.info(f"{ticker}: Found {len(filings)} filings")
                
                # Process each filing
                for filing in filings:
                    # Download filing
                    file_path = await self.sec_fetcher.download_filing(filing)
                    
                    if not file_path:
                        continue
                    
                    stats['filings_downloaded'] += 1
                    
                    # Extract text and sections
                    sections = self.sec_fetcher.extract_text_from_filing(file_path)
                    
                    # Create documents
                    documents = []
                    
                    for section_name, section_text in sections.items():
                        # Create metadata
                        metadata = DocumentMetadata(
                            doc_id=f"{ticker}_{filing['filing_type']}_{filing['filing_date']}_{section_name}",
                            source_type='sec_filing',
                            company_name=ticker,
                            ticker=ticker,
                            filing_type=filing['filing_type'],
                            fiscal_year=int(filing['filing_date'][:4]),
                            url=filing['url'],
                            file_path=str(file_path),
                            document_date=filing['filing_date'],
                            section=section_name
                        )
                        
                        # Clean text
                        cleaned_text = self.doc_processor.clean_text(section_text)
                        
                        doc = Document(
                            page_content=cleaned_text,
                            metadata=metadata.to_dict()
                        )
                        
                        documents.append(doc)
                    
                    stats['documents_created'] += len(documents)
                    
                    # Chunk documents
                    chunked_docs = self.doc_processor.chunk_documents(
                        documents,
                        strategy='semantic'
                    )
                    
                    # Filter low-quality chunks
                    filtered_docs = self.doc_filter.filter_by_length(
                        chunked_docs,
                        min_length=100
                    )
                    
                    # Add to vector store
                    added = await self.vector_store.add_documents_batch(filtered_docs)
                    stats['chunks_indexed'] += added
                    
                    logger.info(
                        f"{ticker} {filing['filing_type']} {filing['filing_date']}: "
                        f"Indexed {added} chunks"
                    )
                
                stats['tickers_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                stats['errors'].append({'ticker': ticker, 'error': str(e)})
        
        # Update global stats
        self.ingestion_stats['sec_filings'] += stats['chunks_indexed']
        self.ingestion_stats['total_documents'] += stats['chunks_indexed']
        self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
        if progress_callback:
            progress_callback(None, 1.0, "Ingestion complete")
        
        logger.info(f"SEC ingestion complete: {stats}")
        
        return stats
    
    async def ingest_pdfs_from_directory(
        self,
        directory: Path,
        source_type: str = 'financial_report',
        recursive: bool = True,
        metadata_extractor: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest all PDFs from a directory
        
        Args:
            directory: Directory path
            source_type: Document source type
            recursive: Search subdirectories
            metadata_extractor: Function to extract metadata from filename
            progress_callback: Progress callback
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting PDF ingestion from {directory}")
        
        stats = {
            'pdfs_found': 0,
            'pdfs_processed': 0,
            'documents_created': 0,
            'chunks_indexed': 0,
            'errors': []
        }
        
        # Find all PDFs
        if recursive:
            pdf_files = list(directory.rglob('*.pdf'))
        else:
            pdf_files = list(directory.glob('*.pdf'))
        
        stats['pdfs_found'] = len(pdf_files)
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                if progress_callback:
                    progress_callback(
                        pdf_path.name,
                        i / len(pdf_files),
                        f"Processing {pdf_path.name}"
                    )
                
                # Extract metadata from filename
                if metadata_extractor:
                    file_metadata = metadata_extractor(pdf_path.name)
                else:
                    file_metadata = self.pdf_processor.extract_metadata_from_filename(
                        pdf_path.name
                    )
                
                # Process PDF
                pdf_data = self.pdf_processor.process_pdf(pdf_path)
                
                if not pdf_data['success']:
                    stats['errors'].append({
                        'file': pdf_path.name,
                        'error': pdf_data.get('error', 'Unknown error')
                    })
                    continue
                
                stats['pdfs_processed'] += 1
                
                # Create documents from pages
                documents = []
                
                for page_data in pdf_data['pages']:
                    metadata = DocumentMetadata(
                        doc_id=f"{pdf_path.stem}_page_{page_data['page_number']}",
                        source_type=source_type,
                        file_path=str(pdf_path),
                        page_number=page_data['page_number'],
                        **file_metadata
                    )
                    
                    # Clean text
                    cleaned_text = self.doc_processor.clean_text(page_data['text'])
                    
                    doc = Document(
                        page_content=cleaned_text,
                        metadata=metadata.to_dict()
                    )
                    
                    documents.append(doc)
                
                stats['documents_created'] += len(documents)
                
                # Chunk documents
                chunked_docs = self.doc_processor.chunk_documents(
                    documents,
                    strategy='recursive'
                )
                
                # Filter and deduplicate
                filtered_docs = self.doc_filter.filter_by_length(
                    chunked_docs,
                    min_length=100
                )
                
                unique_docs = self.doc_deduplicator.deduplicate(filtered_docs)
                
                # Add to vector store
                added = await self.vector_store.add_documents_batch(unique_docs)
                stats['chunks_indexed'] += added
                
                logger.info(f"{pdf_path.name}: Indexed {added} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                stats['errors'].append({'file': pdf_path.name, 'error': str(e)})
        
        # Update global stats
        self.ingestion_stats['pdfs'] += stats['chunks_indexed']
        self.ingestion_stats['total_documents'] += stats['chunks_indexed']
        self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
        if progress_callback:
            progress_callback(None, 1.0, "PDF ingestion complete")
        
        logger.info(f"PDF ingestion complete: {stats}")
        
        return stats
    
    async def ingest_urls(
        self,
        urls: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Ingest content from URLs
        
        Args:
            urls: List of URL dictionaries with metadata
            progress_callback: Progress callback
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting URL ingestion for {len(urls)} URLs")
        
        stats = {
            'urls_processed': 0,
            'documents_created': 0,
            'chunks_indexed': 0,
            'errors': []
        }
        
        for i, url_data in enumerate(urls):
            try:
                url = url_data['url']
                
                if progress_callback:
                    progress_callback(url, i / len(urls), f"Scraping {url}")
                
                # Scrape URL
                scraped_data = await self.url_scraper.scrape_url(url, use_cache=False)
                
                if scraped_data['status'] != 'success':
                    stats['errors'].append({
                        'url': url,
                        'error': scraped_data.get('error', 'Unknown error')
                    })
                    continue
                
                stats['urls_processed'] += 1
                
                # Create metadata
                metadata = DocumentMetadata(
                    doc_id=scraped_data['metadata'].get('url'),
                    source_type=url_data.get('source_type', 'web_article'),
                    url=url,
                    company_name=url_data.get('company_name'),
                    ticker=url_data.get('ticker'),
                    document_date=scraped_data['scraped_at']
                )
                
                # Update metadata with scraped info
                metadata_dict = metadata.to_dict()
                metadata_dict.update(scraped_data['metadata'])
                
                # Clean text
                cleaned_text = self.doc_processor.clean_text(scraped_data['content'])
                
                doc = Document(
                    page_content=cleaned_text,
                    metadata=metadata_dict
                )
                
                stats['documents_created'] += 1
                
                # Chunk document
                chunked_docs = self.doc_processor.chunk_documents(
                    [doc],
                    strategy='recursive'
                )
                
                # Filter
                filtered_docs = self.doc_filter.filter_by_length(
                    chunked_docs,
                    min_length=100
                )
                
                # Add to vector store
                added = await self.vector_store.add_documents_batch(filtered_docs)
                stats['chunks_indexed'] += added
                
                logger.info(f"{url}: Indexed {added} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {url_data.get('url')}: {e}")
                stats['errors'].append({
                    'url': url_data.get('url'),
                    'error': str(e)
                })
        
        # Update global stats
        self.ingestion_stats['urls'] += stats['chunks_indexed']
        self.ingestion_stats['total_documents'] += stats['chunks_indexed']
        self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()
        
        if progress_callback:
            progress_callback(None, 1.0, "URL ingestion complete")
        
        logger.info(f"URL ingestion complete: {stats}")
        
        return stats
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def query(
        self,
        question: str,
        retrieval_strategy: str = "hybrid",
        filter_dict: Optional[Dict[str, Any]] = None,
        num_sources: int = 5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            retrieval_strategy: 'hybrid', 'vector', 'multi_query', 'compression'
            filter_dict: Metadata filters
            num_sources: Number of sources to retrieve
            use_reranking: Apply re-ranking to results
            
        Returns:
            Answer with sources and metadata
        """
        try:
            # Retrieve documents based on strategy
            if retrieval_strategy == "hybrid":
                results = self.hybrid_search.hybrid_search(
                    query=question,
                    filter_dict=filter_dict,
                    limit=num_sources * 2 if use_reranking else num_sources
                )
            elif retrieval_strategy == "multi_query":
                results = self.multi_query_retriever.retrieve(
                    query=question,
                    filter_dict=filter_dict,
                    num_queries=3,
                    limit=num_sources * 2 if use_reranking else num_sources
                )
            elif retrieval_strategy == "compression":
                results = self.compression_retriever.retrieve(
                    query=question,
                    filter_dict=filter_dict,
                    limit=num_sources * 2 if use_reranking else num_sources
                )
            else:  # Default to vector search
                results = self.vector_store.search(
                    query=question,
                    filter_dict=filter_dict,
                    limit=num_sources * 2 if use_reranking else num_sources
                )
            
            if not results:
                return {
                    'answer': 'No relevant information found in the knowledge base.',
                    'sources': [],
                    'confidence': 0.0,
                    'retrieval_strategy': retrieval_strategy,
                    'num_sources_used': 0
                }
            
            # Apply re-ranking if requested
            if use_reranking:
                results = self.reranking_retriever.rerank(
                    query=question,
                    results=results,
                    top_k=num_sources
                )
            
            # Build context
            context = self._build_context(results)
            
            # Generate answer
            answer = self._generate_answer(question, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            return {
                'answer': answer,
                'sources': self._format_sources(results),
                'confidence': confidence,
                'retrieval_strategy': retrieval_strategy,
                'num_sources_used': len(results)
            }
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'error': str(e),
                'answer': 'An error occurred while processing your query.',
                'sources': [],
                'num_sources_used': 0,
                'confidence': 0.0,
                'retrieval_strategy': retrieval_strategy
            }
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from search results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source_info = f"Source {i}"
            
            # Add metadata info
            metadata = result.get('metadata', {})
            if metadata.get('ticker'):
                source_info += f" ({metadata['ticker']}"
                if metadata.get('filing_type'):
                    source_info += f", {metadata['filing_type']}"
                if metadata.get('fiscal_year'):
                    source_info += f", {metadata['fiscal_year']}"
                source_info += ")"
            
            context_parts.append(f"{source_info}:\n{result['text']}\n")
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""Based on the following information from financial documents, answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Provide a detailed, accurate answer based solely on the context provided
- Cite specific sources when making claims (e.g., "According to Source 1...")
- If the context doesn't contain enough information, acknowledge this
- Be objective and factual
- Include relevant numbers, dates, and specifics when available
- If comparing companies or time periods, be explicit about the differences

Answer:"""
        
        try:
            answer = self.llm.invoke(prompt)
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating the answer."
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not results:
            return 0.0
        
        # Get scores
        scores = [r.get('score', 0) for r in results]
        
        # Calculate metrics
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        score_variance = np.var(scores) if len(scores) > 1 else 0
        
        # Confidence calculation
        # - High average score = good
        # - High max score = at least one very relevant result
        # - Low variance = consistent relevance
        
        confidence = (
            0.5 * avg_score +
            0.3 * max_score +
            0.2 * (1 - min(score_variance, 1))
        )
        
        return min(confidence, 1.0)
    
    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display"""
        formatted = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            source = {
                'text': result['text'][:500] + '...',
                'relevance_score': result.get('score', 0),
                'source_type': metadata.get('source_type', 'unknown'),
                'ticker': metadata.get('ticker'),
                'company_name': metadata.get('company_name'),
                'filing_type': metadata.get('filing_type'),
                'fiscal_year': metadata.get('fiscal_year'),
                'section': metadata.get('section'),
                'url': metadata.get('url'),
                'document_date': metadata.get('document_date')
            }
            
            # Remove None values
            source = {k: v for k, v in source.items() if v is not None}
            
            formatted.append(source)
        
        return formatted
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        collection_info = self.vector_store.get_collection_info()
        
        return {
            'ingestion_stats': self.ingestion_stats,
            'collection_info': collection_info,
            'retrieval_strategies': [
                'hybrid',
                'vector',
                'multi_query',
                'compression'
            ],
            'supported_sources': ['sec_filings', 'pdfs', 'urls']
        }
    
    def find_similar_partnerships(
        self,
        company1_id: int,
        company2_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical partnerships
        
        Args:
            company1_id: First company ID
            company2_id: Second company ID
            limit: Number of results
            
        Returns:
            List of similar partnership analyses
        """
        # Get company profiles
        company1 = self.db.get_company_profile(company1_id)
        company2 = self.db.get_company_profile(company2_id)
        
        if not company1 or not company2:
            return []
        
        # Build query
        query = f"""
        Find partnerships and analyses similar to:
        Company A: {company1.name} in {company1.industry} (Revenue: ${company1.revenue/1e9:.1f}B)
        Company B: {company2.name} in {company2.industry} (Revenue: ${company2.revenue/1e9:.1f}B)
        
        Focus on partnerships with similar:
        - Industry combinations
        - Company sizes
        - Revenue ranges
        - Strategic objectives
        """
        
        # Search with filters
        filter_dict = {
            'source_type': ['sec_filing', 'research_report', 'case_study']
        }
        
        results = self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            limit=limit
        )
        
        return results
    
    def delete_documents_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """
        Delete documents matching filter criteria
        
        Useful for removing outdated data
        """
        return self.vector_store.delete_by_filter(filter_dict)
    
    def export_knowledge_base(
        self,
        output_file: Path,
        filter_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Export knowledge base to JSON
        
        Useful for backup or analysis
        """
        all_points = []
        next_offset = None
        
        while True:
            points, next_offset = self.vector_store.scroll_points(
                filter_dict=filter_dict,
                limit=100,
                offset=next_offset
            )
            
            all_points.extend(points)
            
            if next_offset is None:
                break
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(all_points, f, indent=2)
        
        logger.info(f"Exported {len(all_points)} documents to {output_file}")
