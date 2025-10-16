"""
Direct SEC filings ingestion using FREE SEC EDGAR API
NO PAID API NEEDED - Uses official SEC endpoints
"""

import asyncio
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

# Load environment
load_dotenv()

from rag_system.document_processors import AdvancedDocumentProcessor, DocumentFilter
from rag_system.vector_store_free import ChromaVectorStore
from rag_system.data_fetchers import DocumentMetadata
from langchain.docstore.document import Document

# SEC Headers (REQUIRED by SEC policy)
SEC_HEADERS = {
    'User-Agent': 'Vaibhav vaibhavpratap630@gmail.com',  # Use YOUR email
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'www.sec.gov'
}

def get_company_cik_mapping():
    """Map of tickers to CIK numbers"""
    return {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "NVDA": "0001045810",
        "META": "0001326801",
        "TSLA": "0001318605",
    }

def fetch_company_filings(ticker, cik, filing_types=['10-K'], max_results=1):
    """
    Fetch filings using FREE SEC EDGAR API
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    # Update Host header for data.sec.gov
    headers = SEC_HEADERS.copy()
    headers['Host'] = 'data.sec.gov'
    
    try:
        time.sleep(0.12)  # Rate limiting (10 req/sec max)
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        recent_filings = data.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            return []
        
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        primary_documents = recent_filings.get('primaryDocument', [])
        
        filings = []
        
        for i in range(len(forms)):
            form = forms[i]
            
            if form not in filing_types:
                continue
            
            if len(filings) >= max_results:
                break
            
            accession = accession_numbers[i].replace('-', '')
            primary_doc = primary_documents[i]
            
            # Construct document URL
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"
            
            filing_info = {
                'ticker': ticker,
                'cik': cik,
                'filing_type': form,
                'filing_date': filing_dates[i],
                'accession_number': accession_numbers[i],
                'document_url': doc_url,
                'company_name': data.get('name', ticker)
            }
            
            filings.append(filing_info)
        
        return filings
    
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return []

def download_filing(filing, cache_dir):
    """
    Download the actual 10-K document
    """
    ticker = filing['ticker']
    filename = f"{ticker}_{filing['filing_type']}_{filing['filing_date']}.html"
    cache_path = cache_dir / filename
    
    if cache_path.exists():
        print(f"   ✅ Using cached: {filename}")
        return cache_path
    
    try:
        time.sleep(0.12)  # Rate limiting
        
        response = requests.get(filing['document_url'], headers=SEC_HEADERS, timeout=30)
        response.raise_for_status()
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"   ✅ Downloaded: {filename} ({len(response.text)} bytes)")
        return cache_path
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None

def extract_sections_from_html(file_path):
    """
    Extract text sections from 10-K HTML
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts, styles, tables
        for element in soup(['script', 'style', 'table']):
            element.decompose()
        
        # Get all text
        text = soup.get_text(separator='\n', strip=True)
        
        # Try to identify sections by ITEM patterns
        sections = {}
        
        # Common 10-K section patterns
        section_patterns = [
            (r'ITEM\s+1[.\s:]+BUSINESS', 'Business'),
            (r'ITEM\s+1A[.\s:]+RISK\s+FACTORS', 'Risk Factors'),
            (r'ITEM\s+7[.\s:]+MANAGEMENT.?S?\s+DISCUSSION', 'Management Discussion'),
            (r'ITEM\s+7A[.\s:]+QUANTITATIVE', 'Quantitative Disclosures'),
        ]
        
        # Find section positions
        matches = []
        for pattern, section_name in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matches.append((section_name, match.start()))
        
        # Sort by position
        matches.sort(key=lambda x: x[1])
        
        # Extract text between sections
        for i, (section_name, start) in enumerate(matches):
            if i + 1 < len(matches):
                end = matches[i + 1][1]
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            
            # Only keep substantial sections
            if len(section_text) > 1000:
                sections[section_name] = section_text
        
        # If no sections found, use entire document
        if not sections:
            sections['Full Document'] = text
        
        print(f"   ✅ Extracted {len(sections)} section(s)")
        return sections
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return {}

async def ingest_sec_filings():
    """
    Main ingestion function using FREE SEC API
    """
    
    print("="*60)
    print("SEC Filings → ChromaDB (FREE SEC EDGAR API)")
    print("="*60)
    
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'NVDA']  # Add more as needed
    FILING_TYPES = ['10-K']
    COUNT_PER_TYPE = 1
    
    print(f"\n📋 Configuration:")
    print(f"   Tickers: {', '.join(TICKERS)}")
    print(f"   Filing Types: {', '.join(FILING_TYPES)}")
    print(f"   API: FREE SEC EDGAR (no API key needed!)")
    
    # Create cache directory
    cache_dir = Path("./data/sec_filings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("\n🔄 Initializing components...")
    
    cik_mapping = get_company_cik_mapping()
    doc_processor = AdvancedDocumentProcessor()
    doc_filter = DocumentFilter()
    
    vector_store = ChromaVectorStore(
        collection_name="nexus_sphere_knowledge",
        persist_directory="./chroma_db"
    )
    
    print("✅ Components initialized")
    
    # Statistics
    stats = {
        'tickers_processed': 0,
        'filings_downloaded': 0,
        'documents_created': 0,
        'chunks_indexed': 0,
        'errors': []
    }
    
    # Process each ticker
    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"Processing: {ticker}")
        print('='*60)
        
        if ticker not in cik_mapping:
            print(f"   ⚠️ No CIK mapping for {ticker}")
            continue
        
        cik = cik_mapping[ticker]
        
        try:
            # Fetch filings
            print(f"\n1️⃣ Fetching {ticker} filings from SEC EDGAR...")
            filings = fetch_company_filings(ticker, cik, FILING_TYPES, COUNT_PER_TYPE)
            
            if not filings:
                print(f"   ⚠️ No filings found")
                stats['errors'].append({'ticker': ticker, 'error': 'No filings found'})
                continue
            
            print(f"   ✅ Found {len(filings)} filing(s)")
            
            # Process each filing
            for filing_idx, filing in enumerate(filings, 1):
                print(f"\n2️⃣ Processing filing {filing_idx}/{len(filings)}")
                print(f"   Type: {filing['filing_type']}")
                print(f"   Date: {filing['filing_date']}")
                
                # Download
                print(f"   📥 Downloading...")
                file_path = download_filing(filing, cache_dir)
                
                if not file_path:
                    continue
                
                stats['filings_downloaded'] += 1
                
                # Extract sections
                print(f"   📄 Extracting sections...")
                sections = extract_sections_from_html(file_path)
                
                if not sections:
                    print(f"   ⚠️ No content extracted")
                    continue
                
                # Create documents
                documents = []
                
                for section_name, section_text in sections.items():
                    metadata = DocumentMetadata(
                        doc_id=f"{ticker}_{filing['filing_type']}_{filing['filing_date']}_{section_name}",
                        source_type='sec_filing',
                        company_name=filing['company_name'],
                        ticker=ticker,
                        filing_type=filing['filing_type'],
                        fiscal_year=int(filing['filing_date'][:4]),
                        document_date=filing['filing_date'],
                        section=section_name
                    )
                    
                    cleaned_text = doc_processor.clean_text(section_text)
                    
                    doc = Document(
                        page_content=cleaned_text,
                        metadata=metadata.to_dict()
                    )
                    
                    documents.append(doc)
                
                stats['documents_created'] += len(documents)
                print(f"   ✅ Created {len(documents)} document(s)")
                
                # Chunk
                print(f"   ✂️ Chunking...")
                chunked_docs = doc_processor.chunk_documents(documents, strategy='semantic')
                print(f"   ✅ Created {len(chunked_docs)} chunk(s)")
                
                # Filter
                filtered_docs = doc_filter.filter_by_length(chunked_docs, min_length=100)
                print(f"   🔍 Filtered to {len(filtered_docs)} chunks")
                
                # Add to ChromaDB
                print(f"   💾 Adding to ChromaDB...")
                added = await vector_store.add_documents_batch(filtered_docs)
                stats['chunks_indexed'] += added
                
                print(f"   ✅ Indexed {added} chunks")
            
            stats['tickers_processed'] += 1
            print(f"\n✅ {ticker} complete!")
            
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            stats['errors'].append({'ticker': ticker, 'error': str(e)})
    
    # Summary
    print("\n" + "="*60)
    print("📊 INGESTION COMPLETE")
    print("="*60)
    print(f"\n✅ Tickers Processed: {stats['tickers_processed']}/{len(TICKERS)}")
    print(f"✅ Filings Downloaded: {stats['filings_downloaded']}")
    print(f"✅ Documents Created: {stats['documents_created']}")
    print(f"✅ Chunks Indexed: {stats['chunks_indexed']}")
    
    if stats['errors']:
        print(f"\n⚠️ Errors: {len(stats['errors'])}")
        for error in stats['errors']:
            print(f"   - {error['ticker']}: {error['error']}")
    
    # Verify
    info = vector_store.get_collection_info()
    print(f"\n✅ Total documents in ChromaDB: {info['points_count']}")
    
    # Test search
    if info['points_count'] > 0:
        print(f"\n🔎 Testing search...")
        results = vector_store.search("What does Apple do?", limit=3)
        
        if results:
            print(f"   ✅ Search works! Found {len(results)} results")
            print(f"   Top result: {results[0]['text'][:100]}...")
    
    print("\n🎉 Done! Open Streamlit: streamlit run main.py")

if __name__ == "__main__":
    asyncio.run(ingest_sec_filings())