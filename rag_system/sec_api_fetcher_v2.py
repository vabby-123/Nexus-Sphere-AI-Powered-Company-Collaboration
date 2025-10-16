"""
SEC Fetcher using sec-api.io with Extractor API
Gets clean, structured text directly - no HTML parsing needed!
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json

from sec_api import QueryApi, ExtractorApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecApiExtractorFetcher:
    """
    SEC Fetcher using sec-api.io's Extractor API
    
    Much cleaner - gets structured sections directly!
    """
    
    def __init__(self, api_key: str, cache_dir: str = "./data/sec_filings"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize APIs
        self.query_api = QueryApi(api_key=api_key)
        self.extractor_api = ExtractorApi(api_key=api_key)
        
        logger.info("✅ Initialized sec-api.io with Extractor API")
    
    async def fetch_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K'],
        count: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch filings - same as before"""
        all_filings = []
        
        for filing_type in filing_types:
            try:
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
                
                logger.info(f"Querying for {ticker} {filing_type}...")
                
                response = await asyncio.to_thread(
                    self.query_api.get_filings,
                    query
                )
                
                filings = response.get('filings', [])
                logger.info(f"Found {len(filings)} {filing_type} filings")
                
                for filing in filings:
                    filing_date = filing.get('filedAt', '')[:10]
                    
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
                        'company_name': filing.get('companyName')
                    }
                    
                    all_filings.append(filing_info)
            
            except Exception as e:
                logger.error(f"Error fetching {filing_type}: {e}")
        
        return all_filings
    
    async def download_filing(
        self,
        filing: Dict[str, Any],
        force_refresh: bool = False
    ) -> Optional[Path]:
        """
        Extract sections using Extractor API
        
        This is MUCH better - gets clean text directly!
        """
        safe_ticker = filing['ticker'].replace('/', '_')
        filename = f"{safe_ticker}_{filing['filing_type']}_{filing['filing_date']}_{filing['accession_number'].replace('-', '')}.json"
        cache_path = self.cache_dir / filename
        
        if cache_path.exists() and not force_refresh:
            logger.info(f"Using cached extracted data: {filename}")
            return cache_path
        
        try:
            logger.info(f"Extracting {filing['filing_type']} using Extractor API...")
            
            # Use Extractor API to get clean sections
            url = filing['url']
            
            # Extract specific sections
            sections_to_extract = [
                "1",    # Business
                "1A",   # Risk Factors  
                "7",    # MD&A
                "7A",   # Quantitative and Qualitative Disclosures
            ]
            
            extracted_data = {}
            
            for section in sections_to_extract:
                try:
                    section_text = await asyncio.to_thread(
                        self.extractor_api.get_section,
                        url,
                        section,
                        "text"  # Get as plain text, not HTML
                    )
                    
                    if section_text:
                        extracted_data[f"Item {section}"] = section_text
                        logger.info(f"  ✅ Extracted Item {section}: {len(section_text)} chars")
                
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not extract Item {section}: {e}")
            
            if not extracted_data:
                logger.warning("No sections extracted!")
                return None
            
            # Save extracted data
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2)
            
            logger.info(f"Saved extracted sections: {filename}")
            return cache_path
            
        except Exception as e:
            logger.error(f"Error extracting filing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_text_from_filing(self, file_path: Path) -> Dict[str, str]:
        """
        Load extracted sections from JSON
        
        Much simpler - data is already clean!
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)
            
            logger.info(f"Loaded {len(sections)} sections from {file_path.name}")
            return sections
            
        except Exception as e:
            logger.error(f"Error loading sections: {e}")
            return {}