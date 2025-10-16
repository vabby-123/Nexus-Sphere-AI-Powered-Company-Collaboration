"""
Fix SEC fetcher to use more reliable methods
"""

from pathlib import Path
import re

file_path = Path('rag_system/data_fetchers.py')
content = file_path.read_text(encoding='utf-8')

# Backup original
backup_path = Path('rag_system/data_fetchers.py.backup')
backup_path.write_text(content, encoding='utf-8')
print(f"✅ Backed up original to {backup_path}")

# Fix 1: Update get_cik method to use company_tickers.json endpoint
get_cik_new = '''    async def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK string or None if not found
        """
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
        
        # Use SEC's company tickers JSON endpoint (more reliable)
        url = "https://www.sec.gov/files/company_tickers.json"
        
        try:
            # Use requests instead of aiohttp for better reliability
            import requests
            import time
            time.sleep(0.11)  # Rate limiting
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Search for ticker in the data
                for key, company in data.items():
                    if company.get('ticker', '').upper() == ticker:
                        cik = str(company.get('cik_str')).zfill(10)
                        
                        # Cache it
                        self.cik_cache[ticker] = cik
                        self._save_cik_cache()
                        
                        logger.info(f"Found CIK for {ticker}: {cik}")
                        return cik
        
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
        
        return None'''

# Find and replace get_cik method
get_cik_pattern = r'    async def get_cik\(self, ticker: str\) -> Optional\[str\]:.*?return None'
content = re.sub(get_cik_pattern, get_cik_new, content, flags=re.DOTALL)

print("✅ Updated get_cik method")

# Save
file_path.write_text(content, encoding='utf-8')
print(f"✅ Saved changes to {file_path}")
print("\nNow add 'import requests' and 'import time' to the imports if not already there")