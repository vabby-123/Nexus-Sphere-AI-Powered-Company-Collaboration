"""
Data fetching modules for SEC filings, PDFs, and web content
"""
import requests
import time
import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
import re

import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry
import pymupdf
from pypdf import PdfReader
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dataclasses import dataclass

@dataclass
class DocumentMetadata:
    """Rich metadata for document tracking"""
    doc_id: str
    source_type: str  # 'sec_filing', 'pdf', 'url', 'earnings_call', 'research_report'
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    ingestion_date: str = None
    document_date: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    word_count: Optional[int] = None
    language: str = 'en'
    author: Optional[str] = None
    
    def __post_init__(self):
        if self.ingestion_date is None:
            self.ingestion_date = datetime.now().isoformat()
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class SECFilingsFetcher:
    """
    Fetch and process SEC EDGAR filings
    
    Features:
    - Automatic CIK lookup
    - Rate limiting compliance (10 req/sec)
    - Intelligent caching
    - Section extraction
    - Multiple filing types support
    """
    
    BASE_URL = "https://www.sec.gov"
    RATE_LIMIT = 10  # requests per second
    
    def __init__(self, email: str, cache_dir: str = "./data/sec_filings"):
        """
        Initialize SEC fetcher
        
        Args:
            email: Your email (required by SEC)
            cache_dir: Directory for caching downloaded filings
        """
        self.headers = {
            'User-Agent': f'NexusSphere/1.0 ({email})',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CIK cache
        self.cik_cache = {}
        self._load_cik_cache()
    
    def _load_cik_cache(self):
        """Load CIK cache from disk"""
        cache_file = self.cache_dir / 'cik_cache.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.cik_cache = json.load(f)
    
    def _save_cik_cache(self):
        """Save CIK cache to disk"""
        cache_file = self.cache_dir / 'cik_cache.json'
        with open(cache_file, 'w') as f:
            json.dump(self.cik_cache, f, indent=2)
    
    @sleep_and_retry
    @limits(calls=RATE_LIMIT, period=1)
    async def _rate_limited_request(self, session: aiohttp.ClientSession, url: str, **kwargs):
        """Rate-limited async request"""
        async with session.get(url, headers=self.headers, **kwargs) as response:
            return response
    
    async def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a ticker"""
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
        
        # Use SEC's company tickers JSON endpoint
        url = "https://www.sec.gov/files/company_tickers.json"
        
        try:
            time.sleep(0.11)  # Rate limiting
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for key, company in data.items():
                    if company.get('ticker', '').upper() == ticker:
                        cik = str(company.get('cik_str')).zfill(10)
                        self.cik_cache[ticker] = cik
                        self._save_cik_cache()
                        logger.info(f"Found CIK for {ticker}: {cik}")
                        return cik
        
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
        
        return None
    
    async def fetch_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K', '10-Q', '8-K'],
        count: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent filings for a company"""
        cik = await self.get_cik(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return []
        
        all_filings = []
        
        # FIXED: SEC wants CIK without leading zeros for this endpoint
        cik_no_zeros = cik.lstrip('0')
        url = f"https://data.sec.gov/submissions/CIK{cik_no_zeros}.json"
        
        logger.info(f"Fetching from: {url}")
        
        try:
            time.sleep(0.12)  # Rate limiting
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get recent filings
                recent_filings = data.get('filings', {}).get('recent', {})
                
                if not recent_filings:
                    logger.warning(f"No recent filings found for {ticker}")
                    return []
                
                # Extract filing information
                forms = recent_filings.get('form', [])
                filing_dates = recent_filings.get('filingDate', [])
                accession_numbers = recent_filings.get('accessionNumber', [])
                primary_documents = recent_filings.get('primaryDocument', [])
                
                # Count filings by type
                type_counts = {ft: 0 for ft in filing_types}
                
                # Process each filing
                for i in range(len(forms)):
                    form = forms[i]
                    
                    # Check if this is a filing type we want
                    if form not in filing_types:
                        continue
                    
                    # Check if we've hit the limit for this type
                    if type_counts[form] >= count:
                        continue
                    
                    # Check date range
                    filing_date = filing_dates[i]
                    if start_date and filing_date < start_date:
                        continue
                    if end_date and filing_date > end_date:
                        continue
                    
                    # Build URLs
                    accession = accession_numbers[i].replace('-', '')
                    primary_doc = primary_documents[i]
                    
                    filing_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_numbers[i]}&xbrl_type=v"
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession}/{primary_doc}"
                    
                    filing_info = {
                        'ticker': ticker,
                        'cik': cik,
                        'filing_type': form,
                        'filing_date': filing_date,
                        'accession_number': accession_numbers[i],
                        'url': filing_url,
                        'document_url': document_url
                    }
                    
                    all_filings.append(filing_info)
                    type_counts[form] += 1
                
                logger.info(f"Found {len(all_filings)} filings for {ticker}")
            else:
                logger.error(f"SEC API returned status {response.status_code} for {ticker}")
                logger.error(f"URL: {url}")
        
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            import traceback
            traceback.print_exc()
        
        return all_filings
    
    def _get_document_url(self, filing_href: str) -> str:
        """Extract primary document URL from filing page URL"""
        # The filing_href points to the index page
        # We need to extract the actual document
        # This is a simplified version - in production, parse the index page
        return filing_href.replace('-index.html', '.txt')
    
    async def download_filing(self, filing: Dict[str, Any], force_refresh: bool = False) -> Optional[Path]:
        """Download filing to cache"""
        safe_ticker = filing['ticker'].replace('/', '_')
        filename = f"{safe_ticker}_{filing['filing_type']}_{filing['filing_date']}_{filing['accession_number'].replace('-', '')}.html"
        cache_path = self.cache_dir / filename
        
        # Check cache
        if cache_path.exists() and not force_refresh:
            logger.info(f"Using cached filing: {filename}")
            return cache_path
        
        try:
            time.sleep(0.12)  # Rate limiting (~8 req/sec)
            response = requests.get(filing['document_url'], headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Downloaded: {filename}")
                return cache_path
            else:
                logger.error(f"Failed to download: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
        
        return None
    
    def extract_text_from_filing(self, file_path: Path) -> Dict[str, str]:
        """
        Extract text and identify sections from SEC filing
        
        Args:
            file_path: Path to filing file
            
        Returns:
            Dictionary mapping section names to text content
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return {}
        
        # Remove HTML tags
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove tables (often financial data that's hard to parse)
        for table in soup.find_all('table'):
            table.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        # Identify sections
        sections = self._identify_sec_sections(text)
        
        return sections
    
    def _identify_sec_sections(self, text: str) -> Dict[str, str]:
        """
        Identify standard SEC filing sections using regex
        
        Handles both 10-K and 10-Q sections
        """
        # Section patterns for 10-K
        section_patterns_10k = [
            ('Business', r'ITEM\s+1[.\s:]+BUSINESS'),
            ('Risk Factors', r'ITEM\s+1A[.\s:]+RISK\s+FACTORS'),
            ('Unresolved Staff Comments', r'ITEM\s+1B[.\s:]+UNRESOLVED\s+STAFF'),
            ('Properties', r'ITEM\s+2[.\s:]+PROPERTIES'),
            ('Legal Proceedings', r'ITEM\s+3[.\s:]+LEGAL\s+PROCEEDINGS'),
            ('Mine Safety', r'ITEM\s+4[.\s:]+MINE\s+SAFETY'),
            ('Market for Stock', r'ITEM\s+5[.\s:]+MARKET\s+FOR.*STOCK'),
            ('Selected Financial Data', r'ITEM\s+6[.\s:]+SELECTED\s+FINANCIAL'),
            ('Management Discussion', r'ITEM\s+7[.\s:]+MANAGEMENT.?S?\s+DISCUSSION'),
            ('Quantitative Qualitative Disclosures', r'ITEM\s+7A[.\s:]+QUANTITATIVE\s+AND\s+QUALITATIVE'),
            ('Financial Statements', r'ITEM\s+8[.\s:]+FINANCIAL\s+STATEMENTS'),
            ('Changes in Disagreements', r'ITEM\s+9[.\s:]+CHANGES\s+IN\s+AND\s+DISAGREEMENTS'),
            ('Controls and Procedures', r'ITEM\s+9A[.\s:]+CONTROLS\s+AND\s+PROCEDURES'),
            ('Other Information', r'ITEM\s+9B[.\s:]+OTHER\s+INFORMATION'),
            ('Directors and Officers', r'ITEM\s+10[.\s:]+DIRECTORS.*OFFICERS'),
            ('Executive Compensation', r'ITEM\s+11[.\s:]+EXECUTIVE\s+COMPENSATION'),
            ('Security Ownership', r'ITEM\s+12[.\s:]+SECURITY\s+OWNERSHIP'),
            ('Certain Relationships', r'ITEM\s+13[.\s:]+CERTAIN\s+RELATIONSHIPS'),
            ('Principal Accountant', r'ITEM\s+14[.\s:]+PRINCIPAL\s+ACCOUNTANT'),
            ('Exhibits', r'ITEM\s+15[.\s:]+EXHIBITS'),
        ]
        
        # Section patterns for 10-Q
        section_patterns_10q = [
            ('Financial Statements', r'PART\s+I.*ITEM\s+1[.\s:]+FINANCIAL\s+STATEMENTS'),
            ('Management Discussion', r'PART\s+I.*ITEM\s+2[.\s:]+MANAGEMENT.?S?\s+DISCUSSION'),
            ('Quantitative Qualitative Disclosures', r'PART\s+I.*ITEM\s+3[.\s:]+QUANTITATIVE\s+AND\s+QUALITATIVE'),
            ('Controls and Procedures', r'PART\s+I.*ITEM\s+4[.\s:]+CONTROLS\s+AND\s+PROCEDURES'),
            ('Legal Proceedings', r'PART\s+II.*ITEM\s+1[.\s:]+LEGAL\s+PROCEEDINGS'),
            ('Risk Factors', r'PART\s+II.*ITEM\s+1A[.\s:]+RISK\s+FACTORS'),
        ]
        
        sections = {}
        
        # Try 10-K patterns first
        section_patterns = section_patterns_10k + section_patterns_10q
        
        # Find all matches
        matches = []
        for section_name, pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                matches.append((section_name, match.start(), match.end()))
        
        # Sort by position
        matches.sort(key=lambda x: x[1])
        
        # Extract text between matches
        for i, (section_name, start, end) in enumerate(matches):
            # Find end position (start of next section or end of document)
            if i + 1 < len(matches):
                next_start = matches[i + 1][1]
            else:
                next_start = len(text)
            
            section_text = text[start:next_start].strip()
            
            # Only add if substantial content
            if len(section_text) > 200:
                sections[section_name] = section_text
        
        # If no sections found, treat as single document
        if not sections:
            sections['Full Document'] = text
        
        logger.info(f"Identified {len(sections)} sections in filing")
        
        return sections
    
    async def batch_download_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = ['10-K', '10-Q'],
        count_per_type: int = 5,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Path]]:
        """
        Batch download filings for multiple companies
        
        Args:
            tickers: List of ticker symbols
            filing_types: Filing types to download
            count_per_type: Number of each filing type
            progress_callback: Callback function(ticker, progress)
            
        Returns:
            Dictionary mapping ticker to list of downloaded file paths
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
            
            # Fetch filings
            filings = await self.fetch_company_filings(
                ticker=ticker,
                filing_types=filing_types,
                count=count_per_type
            )
            
            # Download each filing
            downloaded_files = []
            for filing in filings:
                file_path = await self.download_filing(filing)
                if file_path:
                    downloaded_files.append(file_path)
            
            results[ticker] = downloaded_files
            
            # Progress callback
            if progress_callback:
                progress_callback(ticker, (i + 1) / len(tickers))
            
            logger.info(f"Downloaded {len(downloaded_files)} filings for {ticker}")
        
        return results


class PDFBatchProcessor:
    """
    High-performance PDF processing with parallel extraction
    
    Features:
    - Batch processing
    - Parallel execution
    - Multiple extraction methods (PyMuPDF, pypdf)
    - Image extraction
    - Table detection
    """
    
    def __init__(self, cache_dir: str = "./data/pdf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            max_workers: Number of parallel workers
            progress_callback: Progress callback function
            
        Returns:
            List of processed document metadata
        """
        # Find all PDFs
        if recursive:
            pdf_files = list(directory.rglob('*.pdf'))
        else:
            pdf_files = list(directory.glob('*.pdf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Process in parallel
        results = []
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(pdf_path, index):
            async with semaphore:
                result = await self.process_pdf_async(pdf_path)
                
                if progress_callback:
                    progress_callback(pdf_path.name, (index + 1) / len(pdf_files))
                
                return result
        
        # Process all PDFs
        tasks = [
            process_with_semaphore(pdf_path, i)
            for i, pdf_path in enumerate(pdf_files)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"Successfully processed {len(valid_results)}/{len(pdf_files)} PDFs")
        
        return valid_results
    
    async def process_pdf_async(self, pdf_path: Path) -> Dict[str, Any]:
        """Process single PDF asynchronously"""
        return await asyncio.to_thread(self.process_pdf, pdf_path)
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted data and metadata
        """
        try:
            # Try PyMuPDF first (faster and better)
            doc = pymupdf.open(pdf_path)
            
            pages = []
            metadata = {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'num_pages': len(doc),
                'pdf_metadata': doc.metadata,
                'file_size': pdf_path.stat().st_size,
                'modified_date': datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
            }
            
            for page_num, page in enumerate(doc, start=1):
                # Extract text
                text = page.get_text("text")
                
                # Extract images info (optional)
                images = page.get_images()
                
                page_data = {
                    'page_number': page_num,
                    'text': text,
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'has_images': len(images) > 0,
                    'image_count': len(images)
                }
                
                pages.append(page_data)
            
            doc.close()
            
            logger.info(f"Processed PDF: {pdf_path.name} ({len(pages)} pages)")
            
            return {
                'metadata': metadata,
                'pages': pages,
                'success': True
            }
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            
            # Fallback to pypdf
            try:
                return self._process_with_pypdf(pdf_path)
            except Exception as e2:
                logger.error(f"Fallback also failed for {pdf_path}: {e2}")
                return {
                    'metadata': {'filename': pdf_path.name, 'path': str(pdf_path)},
                    'pages': [],
                    'success': False,
                    'error': str(e)
                }
    
    def _process_with_pypdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Fallback PDF processing with pypdf"""
        reader = PdfReader(pdf_path)
        
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            pages.append({
                'page_number': page_num,
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text)
            })
        
        return {
            'metadata': {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'num_pages': len(reader.pages),
                'pdf_metadata': reader.metadata
            },
            'pages': pages,
            'success': True,
            'processor': 'pypdf'
        }
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename patterns
        
        Common patterns:
        - CompanyName_10K_2023.pdf
        - AAPL_Q4_2023_Earnings.pdf
        - Microsoft_Annual_Report_2023.pdf
        """
        metadata = {}
        
        # Extract year
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            metadata['year'] = int(year_match.group())
        
        # Extract quarter
        quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
        if quarter_match:
            metadata['quarter'] = int(quarter_match.group(1))
        
        # Extract filing type
        filing_types = ['10-K', '10-Q', '8-K', 'DEF 14A', 'S-1']
        for filing_type in filing_types:
            if filing_type.lower() in filename.lower():
                metadata['filing_type'] = filing_type
                break
        
        # Extract ticker (assume 2-5 uppercase letters)
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', filename)
        if ticker_match:
            metadata['ticker'] = ticker_match.group(1)
        
        # Extract document type keywords
        doc_types = {
            'earnings': ['earnings', 'earning'],
            'annual_report': ['annual', 'report'],
            'quarterly': ['quarterly', 'quarter'],
            'presentation': ['presentation', 'deck', 'slides']
        }
        
        for doc_type, keywords in doc_types.items():
            if any(kw in filename.lower() for kw in keywords):
                metadata['document_type'] = doc_type
                break
        
        return metadata


class URLScraper:
    """
    Intelligent web scraping for financial content
    
    Features:
    - Rate limiting
    - User agent rotation
    - JavaScript rendering support
    - Content extraction
    - Metadata parsing
    """
    
    def __init__(self, cache_dir: str = "./data/url_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
    
    async def scrape_url(
        self,
        url: str,
        use_cache: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Scrape content from URL
        
        Args:
            url: URL to scrape
            use_cache: Use cached content if available
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with scraped content and metadata
        """
        # Generate cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                logger.info(f"Using cached content for {url}")
                return cached_data
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': self.user_agents[hash(url) % len(self.user_agents)]
                }
                
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Parse HTML
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract metadata
                        metadata = self._extract_metadata(soup, url)
                        
                        # Extract main content
                        content = self._extract_content(soup)
                        
                        result = {
                            'url': url,
                            'content': content,
                            'metadata': metadata,
                            'scraped_at': datetime.now().isoformat(),
                            'status': 'success'
                        }
                        
                        # Cache result
                        with open(cache_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        logger.info(f"Successfully scraped {url}")
                        
                        return result
                    else:
                        logger.error(f"Failed to scrape {url}: Status {response.status}")
                        return {
                            'url': url,
                            'status': 'error',
                            'error': f"HTTP {response.status}"
                        }
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout scraping {url}")
            return {'url': url, 'status': 'error', 'error': 'Timeout'}
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {'url': url, 'status': 'error', 'error': str(e)}
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {'url': url}
        
        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text(strip=True)
        
        # Meta tags
        meta_tags = {
            'description': ['description', 'og:description'],
            'author': ['author'],
            'publish_date': ['article:published_time', 'datePublished'],
            'keywords': ['keywords']
        }
        
        for key, names in meta_tags.items():
            for name in names:
                meta = soup.find('meta', attrs={'name': name}) or \
                       soup.find('meta', attrs={'property': name})
                if meta:
                    metadata[key] = meta.get('content', '')
                    break
        
        # Structured data (JSON-LD)
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                structured_data = json.loads(json_ld.string)
                metadata['structured_data'] = structured_data
            except:
                pass
        
        return metadata
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        
        # Try to find main content
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_=re.compile('content|article|post', re.I)) or
            soup.find('body')
        )
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    async def scrape_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            progress_callback: Progress callback
            
        Returns:
            List of scraped results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url, index):
            async with semaphore:
                result = await self.scrape_url(url)
                
                if progress_callback:
                    progress_callback(url, (index + 1) / len(urls))
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
                return result
        
        tasks = [scrape_with_semaphore(url, i) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"Successfully scraped {len(valid_results)}/{len(urls)} URLs")
        
        return valid_results