"""
Setup script for Advanced RAG System (NO DOCKER REQUIRED)
"""

import os
import sys
from pathlib import Path
import subprocess

# ============================================================================
# SETUP LOGGING FIRST (before anything else)
# ============================================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    logger.info(f"✅ Loading .env from: {Path('.env').absolute()}")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=True)
    from dotenv import load_dotenv
    load_dotenv(override=True)

# Verify environment variables
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
SEC_EMAIL = os.getenv('SEC_USER_EMAIL')

if GEMINI_KEY:
    logger.info(f"✅ GEMINI_API_KEY loaded: {GEMINI_KEY[:10]}...")
else:
    logger.warning("❌ GEMINI_API_KEY not found in environment")

if SEC_EMAIL:
    logger.info(f"✅ SEC_USER_EMAIL loaded: {SEC_EMAIL}")
else:
    logger.warning("❌ SEC_USER_EMAIL not found in environment")

# ============================================================================
# REST OF THE CODE
# ============================================================================

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        return False
    
    logger.info(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/sec_filings',
        'data/pdfs',
        'data/urls',
        'data/cache',
        'data/temp_uploads',
        'data/exports',
        'chroma_db',
        'faiss_db'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'langchain': 'Core RAG framework',
        'chromadb': 'Vector database (primary)',
        'pymupdf': 'PDF processing',
        'beautifulsoup4': 'Web scraping',
        'aiohttp': 'Async HTTP',
        'tiktoken': 'Token counting'
    }
    
    missing = []
    
    for package, description in required_packages.items():
        try:
            # Handle package names with hyphens
            import_name = package.replace('-', '_')
            if package == 'pymupdf':
                import_name = 'fitz'
            elif package == 'beautifulsoup4':
                import_name = 'bs4'
            
            __import__(import_name)
            logger.info(f"✅ {package}: {description}")
        except ImportError:
            missing.append(package)
            logger.warning(f"⚠️ {package} not found: {description}")
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.info("\n📦 Install with:")
        logger.info("   pip install -r requirements_rag.txt\n")
        return False
    
    return True


def check_vector_stores():
    """Check vector store availability"""
    logger.info("\n" + "="*60)
    logger.info("Checking Vector Stores")
    logger.info("="*60)
    
    stores_available = []
    
    # Check ChromaDB
    try:
        import chromadb
        logger.info("✅ ChromaDB available (PRIMARY - NO DOCKER NEEDED)")
        logger.info("   - Persistent local storage")
        logger.info("   - Metadata filtering")
        logger.info("   - Best for most use cases")
        stores_available.append('chromadb')
    except ImportError:
        logger.warning("⚠️ ChromaDB not available")
    
    # Check FAISS
    try:
        import faiss
        logger.info("✅ FAISS available (ALTERNATIVE)")
        logger.info("   - Fastest similarity search")
        logger.info("   - Lower memory usage")
        logger.info("   - Good for < 1M documents")
        stores_available.append('faiss')
    except ImportError:
        logger.warning("⚠️ FAISS not available")
    
    if not stores_available:
        logger.error("❌ No vector stores available!")
        logger.info("\n📦 Install ChromaDB:")
        logger.info("   pip install chromadb")
        return False
    
    logger.info(f"\n✅ {len(stores_available)} vector store(s) available")
    return True


def check_environment_variables():
    """Check required environment variables"""
    required_vars = {
        'GEMINI_API_KEY': 'Required for embeddings and LLM',
        'SEC_USER_EMAIL': 'Required for SEC EDGAR API'
    }
    
    missing = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        
        if value:
            # Mask API keys
            if 'KEY' in var:
                masked = value[:10] + '...' + value[-4:]
            else:
                masked = value
            
            logger.info(f"✅ {var}: {masked}")
        else:
            missing.append(f"{var}: {description}")
            logger.warning(f"⚠️ {var} not set")
    
    if missing:
        logger.warning("\n⚠️ Missing environment variables:")
        for item in missing:
            logger.warning(f"   - {item}")
        
        logger.info("\n📝 Create .env file:")
        logger.info("   cp .env.example .env")
        logger.info("   # Then edit .env and add your API keys")
        
        return False
    
    return True


def test_basic_functionality():
    """Test basic RAG functionality"""
    logger.info("\n" + "="*60)
    logger.info("Testing Basic Functionality")
    logger.info("="*60)
    
    try:
        from data_fetchers import DocumentMetadata
        from rag_system.document_processors import AdvancedDocumentProcessor
        
        metadata = DocumentMetadata(
            doc_id="test_doc",
            source_type="test"
        )
        logger.info("✅ Metadata creation works")
        
        processor = AdvancedDocumentProcessor()
        logger.info("✅ Document processor initialized")
        
        # Use FREE embeddings instead
        try:
            from vector_store_free import ChromaVectorStore
            
            logger.info("🔄 Testing with FREE HuggingFace embeddings (no API limits)...")
            
            import shutil
            test_dir = Path("./test_chroma_db")
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            vector_store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory="./test_chroma_db"
            )
            logger.info("✅ ChromaDB with FREE embeddings works!")
            
            # Cleanup
            if test_dir.exists():
                shutil.rmtree(test_dir)
        
        except Exception as e:
            logger.error(f"❌ Vector store test failed: {e}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_env_template():
    """Create .env.example template"""
    env_template = """# Nexus Sphere Advanced RAG Configuration
# NO DOCKER REQUIRED!

# Required: Gemini API Key
# Get from: https://ai.google.dev/
GEMINI_API_KEY=your_gemini_api_key_here

# Required: SEC EDGAR API (use your email)
SEC_USER_EMAIL=your-email@example.com

# Optional: Vector Store Selection
VECTOR_STORE_TYPE=chroma  # or "faiss"

# Optional: OpenAI (for AutoGen)
OPENAI_API_KEY=your_openai_key_here
"""
    
    env_file = Path('.env.example')
    
    if not env_file.exists():
        env_file.write_text(env_template)
        logger.info("✅ Created .env.example template")
    
    if not Path('.env').exists():
        logger.warning("\n⚠️ .env file not found")
        logger.info("📝 Create .env file:")
        logger.info("   cp .env.example .env")
        logger.info("   # Edit and add your API keys")


def print_summary():
    """Print setup summary"""
    logger.info("\n" + "="*60)
    logger.info("🎉 SETUP COMPLETE!")
    logger.info("="*60)
    
    logger.info("""
✅ NO DOCKER REQUIRED!

Your RAG system is ready to use with:
- ChromaDB: Persistent local storage (primary)
- FAISS: Fast in-memory search (alternative)

📊 Next Steps:
1. Make sure .env file has your GEMINI_API_KEY
2. Start application: streamlit run main.py
3. Navigate to "RAG Assistant" in sidebar
4. Choose "Ingest Data" tab
5. Start with SEC filings or PDFs

💡 Tips:
- ChromaDB stores data in ./chroma_db/ (survives restarts)
- FAISS stores data in ./faiss_db/ (faster for < 1M docs)
- Switch between them in Settings tab
- No external services needed!

🔗 Resources:
- ChromaDB Docs: https://docs.trychroma.com/
- FAISS Guide: https://github.com/facebookresearch/faiss/wiki
""")


def main():
    """Run all setup checks"""
    logger.info("=" * 60)
    logger.info("Nexus Sphere Advanced RAG Setup")
    logger.info("NO DOCKER REQUIRED!")
    logger.info("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Checking dependencies", check_dependencies),
        ("Checking vector stores", check_vector_stores),
        ("Creating .env template", create_env_template),
        ("Checking environment variables", check_environment_variables),
        ("Testing basic functionality", test_basic_functionality)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"{step_name}...")
        logger.info('='*60)
        
        try:
            result = step_func()
            if result is None:
                result = True
            results.append((step_name, result))
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Setup Summary")
    logger.info("=" * 60)
    
    for step_name, result in results:
        status = "✅" if result else "⚠️"
        logger.info(f"{status} {step_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print_summary()
    else:
        logger.warning("\n⚠️ Setup completed with warnings.")
        logger.info("\n📦 Common fixes:")
        logger.info("   pip install -r requirements_rag.txt")


if __name__ == "__main__":
    main()