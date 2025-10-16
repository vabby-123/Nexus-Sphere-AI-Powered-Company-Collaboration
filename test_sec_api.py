import requests
import json
from datetime import datetime
import time

# Top 10 companies by market cap with their CIK numbers
TOP_COMPANIES = {
    "0000320193": {"ticker": "AAPL", "name": "Apple Inc."},
    "0000789019": {"ticker": "MSFT", "name": "Microsoft Corporation"},
    "0001652044": {"ticker": "GOOGL", "name": "Alphabet Inc."},
    "0001018724": {"ticker": "AMZN", "name": "Amazon.com Inc."},
    "0001045810": {"ticker": "NVDA", "name": "NVIDIA Corporation"},
    "0001326801": {"ticker": "META", "name": "Meta Platforms Inc."},
    "0001318605": {"ticker": "TSLA", "name": "Tesla Inc."},
    "0001067983": {"ticker": "BRK.B", "name": "Berkshire Hathaway Inc."},
    "0001403161": {"ticker": "V", "name": "Visa Inc."},
    "0000019617": {"ticker": "JPM", "name": "JPMorgan Chase & Co."}
}

# Required headers for SEC API (must include User-Agent per SEC policy)
HEADERS = {
    'User-Agent': 'Vaibhav vaibhavpratap630@gmail.com',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

def fetch_company_filings(cik):
    """
    Fetch all filings for a company using SEC's submissions API
    
    Args:
        cik: 10-digit CIK number with leading zeros
    
    Returns:
        JSON data with company filings
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    try:
        # SEC requires a delay between requests (be respectful)
        time.sleep(0.1)
        
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for CIK {cik}: {e}")
        return None

def filter_10k_10q_filings(filings_data, max_results=5):
    """
    Filter and extract 10-K and 10-Q filings from the submissions data
    """
    if not filings_data or 'filings' not in filings_data:
        return []
    
    recent_filings = filings_data['filings']['recent']
    
    filtered_filings = []
    
    # Iterate through all filings
    for i in range(len(recent_filings['form'])):
        form_type = recent_filings['form'][i]
        
        # Only include 10-K and 10-Q forms
        if form_type in ['10-K', '10-Q']:
            filing = {
                'formType': form_type,
                'filingDate': recent_filings['filingDate'][i],
                'reportDate': recent_filings['reportDate'][i],
                'accessionNumber': recent_filings['accessionNumber'][i],
                'primaryDocument': recent_filings['primaryDocument'][i],
                'primaryDocDescription': recent_filings['primaryDocDescription'][i]
            }
            
            # Construct document URL
            accession_no_no_dashes = filing['accessionNumber'].replace('-', '')
            filing['documentUrl'] = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{filings_data['cik']}/{accession_no_no_dashes}/"
                f"{filing['primaryDocument']}"
            )
            
            filtered_filings.append(filing)
            
            if len(filtered_filings) >= max_results:
                break
    
    return filtered_filings

def display_company_filings(cik, company_info, filings):
    """
    Display filings in a readable format
    """
    ticker = company_info['ticker']
    name = company_info['name']
    
    print(f"\n{'='*80}")
    print(f"📊 {name} ({ticker}) - CIK: {cik}")
    print(f"{'='*80}")
    
    if not filings:
        print("No recent 10-K or 10-Q filings found.")
        return
    
    for filing in filings:
        print(f"\n  📄 Form Type: {filing['formType']}")
        print(f"  📅 Filing Date: {filing['filingDate']}")
        print(f"  📋 Report Date: {filing['reportDate']}")
        print(f"  📝 Description: {filing['primaryDocDescription']}")
        print(f"  🔗 Document URL: {filing['documentUrl']}")
        print(f"  📥 Accession Number: {filing['accessionNumber']}")
        print(f"  {'-'*76}")

def download_filing(filing, cik, ticker):
    """
    Download a specific filing document
    """
    try:
        time.sleep(0.1)  # Be respectful to SEC servers
        
        response = requests.get(filing['documentUrl'], headers=HEADERS)
        response.raise_for_status()
        
        # Create filename
        filename = f"{ticker}_{filing['formType']}_{filing['filingDate']}.html"
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading filing: {e}")
        return None

def main():
    """
    Main function to fetch and display filings for top 10 companies
    """
    print("🔍 Fetching 10-K and 10-Q reports for Top 10 Companies from SEC EDGAR...")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: Update the User-Agent in HEADERS with your name and email!")
    print("=" * 80)
    
    all_results = {}
    
    for cik, company_info in TOP_COMPANIES.items():
        ticker = company_info['ticker']
        name = company_info['name']
        
        print(f"\nFetching data for {name} ({ticker})...")
        
        # Fetch all filings data
        filings_data = fetch_company_filings(cik)
        
        if filings_data:
            # Filter for 10-K and 10-Q
            filtered_filings = filter_10k_10q_filings(filings_data, max_results=5)
            
            # Store results
            all_results[ticker] = {
                'company_info': company_info,
                'cik': cik,
                'filings': filtered_filings
            }
            
            # Display results
            display_company_filings(cik, company_info, filtered_filings)
    
    # Save all data to JSON file
    with open('sec_filings_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ Data fetch complete! Summary saved to 'sec_filings_summary.json'")
    print(f"{'='*80}")
    
    # Optional: Download first filing for first company (uncomment to use)
    # first_ticker = list(TOP_COMPANIES.keys())[0]
    # if all_results and first_ticker in all_results:
    #     filings = all_results[first_ticker]['filings']
    #     if filings:
    #         download_filing(filings[0], first_ticker, 
    #                        all_results[first_ticker]['company_info']['ticker'])

if __name__ == "__main__":
    main()