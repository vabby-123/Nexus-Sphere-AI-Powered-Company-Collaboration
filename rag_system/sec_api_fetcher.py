"""
SEC Fetcher using sec-api.io (premium service)
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

from sec_api import QueryApi, RenderApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecApiFetcher:
    """
    SEC Fetcher using sec-api.io premium service
    
    Much more reliable than free EDGAR API!
    """
    
    def __init__(self, api_key: str, cache_dir: str = "./data/sec_filings"):
        """
        Initialize sec-api.io fetcher
        
        Args:
            api_key: Your sec-api.io API key
            cache_dir: Directory for caching downloads
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize APIs
        self.query_api = QueryApi(api_key=api_key)
        self.render_api = RenderApi(api_key=api_key)
        
        logger.info("✅ Initialized sec-api.io fetcher")
    
    async def fetch_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K', '10-Q'],
        count: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch filings using sec-api.io
        
        Args:
            ticker: Stock ticker
            filing_types: List of filing types
            count: Number of filings per type
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of filing metadata
        """
        all_filings = []
        
        for filing_type in filing_types:
            try:
                # Build query
                query = {
                    "query": {
                        "query_string": {
                            "query": f'ticker:{ticker} AND formType:"{filing_type}"'
                        }
                    },
                    "from": "0",
                    "size": str(count),
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
                
                logger.info(f"Querying sec-api.io for {ticker} {filing_type}...")
                
                # Execute query
                response = await asyncio.to_thread(
                    self.query_api.get_filings,
                    query
                )
                
                filings = response.get('filings', [])
                
                logger.info(f"Found {len(filings)} {filing_type} filings for {ticker}")
                
                # Process each filing
                for filing in filings:
                    filing_date = filing.get('filedAt', '')[:10]  # YYYY-MM-DD
                    
                    # Check date range
                    if start_date and filing_date < start_date:
                        continue
                    if end_date and filing_date > end_date:
                        continue
                    
                    filing_info = {
                        'ticker': ticker,
                        'cik': filing.get('cik'),
                        'filing_type': filing.get('formType'),
                        'filing_date': filing_date,
                        'accession_number': filing.get('accessionNo'),
                        'url': filing.get('linkToFilingDetails'),
                        'document_url': filing.get('linkToTxt'),
                        'html_url': filing.get('linkToHtml'),
                        'company_name': filing.get('companyName')
                    }
                    
                    all_filings.append(filing_info)
                
            except Exception as e:
                logger.error(f"Error fetching {filing_type} for {ticker}: {e}")
        
        return all_filings
    
    async def download_filing(
        self,
        filing: Dict[str, Any],
        force_refresh: bool = False
    ) -> Optional[Path]:
        """
        Download filing using sec-api.io's render API
        
        Returns cleaner HTML than raw EDGAR
        """
        # Create cache filename
        safe_ticker = filing['ticker'].replace('/', '_')
        filename = f"{safe_ticker}_{filing['filing_type']}_{filing['filing_date']}_{filing['accession_number'].replace('-', '')}.html"
        cache_path = self.cache_dir / filename
        
        # Check cache
        if cache_path.exists() and not force_refresh:
            logger.info(f"Using cached filing: {filename}")
            return cache_path
        
        try:
            # CRITICAL FIX: Get the primary document URL (the actual 10-K)
            # From the filing metadata, we need the linkToHtml or linkToFilingDetails
            
            # Option 1: Use the HTML URL directly if available
            url = filing.get('linkToHtml') or filing.get('html_url')
            
            # Option 2: If not available, construct it from accession number
            if not url:
                # For Apple: /ix?doc=/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm
                accession = filing['accession_number'].replace('-', '')
                cik = filing['cik']
                
                # Get the primary document name from accession
                # This is usually {ticker}-{date}.htm
                ticker_lower = filing['ticker'].lower()
                date = filing['filing_date'].replace('-', '')
                primary_doc = f"{ticker_lower}-{date}.htm"
                
                url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={filing['accession_number']}&xbrl_type=v"
            
            logger.info(f"Downloading from: {url}")
            
            # Use Render API for clean HTML
            html_content = await asyncio.to_thread(
                self.render_api.get_filing,
                url
            )
            
            # Save to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Downloaded and cached: {filename} ({len(html_content)} bytes)")
            return cache_path
            
        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_text_from_filing(self, file_path: Path) -> Dict[str, str]:
        """
        Extract text from filing
        
        sec-api.io provides cleaner HTML, so extraction is easier
        """
        try:
            from bs4 import BeautifulSoup
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove tables
            for table in soup.find_all('table'):
                table.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            # For now, return as single document
            # You can add section parsing later
            sections = {'Full Document': text}
            
            logger.info(f"Extracted text from {file_path.name}")
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return {}