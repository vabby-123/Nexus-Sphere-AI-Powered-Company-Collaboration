"""
Streamlit UI components for Advanced RAG System
"""

import streamlit as st
import asyncio
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from .advanced_rag import AdvancedRAGSystem


def show_advanced_rag_system():
    """Main UI for Advanced RAG System"""
    
    st.title("🧠 Advanced RAG Knowledge Base")
    if 'vector_store_type' not in st.session_state:
        st.session_state.vector_store_type = 'chroma'
    # Initialize RAG system
    if 'advanced_rag' not in st.session_state:
        with st.spinner("🔄 Initializing Advanced RAG System..."):
            try:
                vector_store_type = st.session_state.vector_store_type
                st.session_state.advanced_rag = AdvancedRAGSystem(
                    db=st.session_state.db,
                    collection_name="nexus_sphere_knowledge",
                    vector_store_type=vector_store_type,  # "chroma" or "faiss"
                    persist_directory=f"./{vector_store_type}_db"
                )
                st.success(f"✅ RAG System initialized with {vector_store_type.upper()}!")
                st.info(f"📁 Data stored in: `./{vector_store_type}_db/`")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")
                return
    
    rag = st.session_state.advanced_rag
    
    # Main tabs
    tabs = st.tabs([
        "💬 Query",
        "📥 Ingest Data",
        "📊 Statistics",
        "🔍 Advanced Search",
        "⚙️ Settings"
    ])
    
    # ========================================================================
    # TAB 1: QUERY INTERFACE
    # ========================================================================
    
    with tabs[0]:
        show_query_interface(rag)
    
    # ========================================================================
    # TAB 2: DATA INGESTION
    # ========================================================================
    
    with tabs[1]:
        show_ingestion_interface(rag)
    
    # ========================================================================
    # TAB 3: STATISTICS
    # ========================================================================
    
    with tabs[2]:
        show_statistics_dashboard(rag)
    
    # ========================================================================
    # TAB 4: ADVANCED SEARCH
    # ========================================================================
    
    with tabs[3]:
        show_advanced_search(rag)
    
    # ========================================================================
    # TAB 5: SETTINGS
    # ========================================================================
    
    with tabs[4]:
        show_settings_panel(rag)


def show_query_interface(rag: AdvancedRAGSystem):
    """Query interface with chat-like experience"""
    
    st.subheader("💬 Query Knowledge Base")
    
    # Example questions
    with st.expander("📝 Example Questions", expanded=False):
        st.markdown("""
        **Financial Analysis:**
        - "What were Apple's main risk factors in their last 10-K filing?"
        - "Compare revenue growth between Microsoft and Google over the past 3 years"
        - "What are the key trends in AI spending from recent 10-K filings?"
        
        **Partnership Intelligence:**
        - "Find partnerships between tech and healthcare companies"
        - "What makes successful cross-industry partnerships?"
        
        **ESG & Sustainability:**
        - "Which companies have carbon neutrality commitments?"
        - "Compare sustainability initiatives across the energy sector"
        
        **Risk Assessment:**
        - "What legal risks do tech companies face in 2024?"
        - "Compare debt levels across retail companies"
        """)
        
        # Quick example buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Revenue Trends", key="ex1"):
                st.session_state.rag_query = "What are the revenue growth trends in the technology sector?"
        with col2:
            if st.button("⚠️ Risk Factors", key="ex2"):
                st.session_state.rag_query = "What are common risk factors mentioned across different industries?"
    
    # Query input
    query = st.text_area(
        "Your Question",
        value=st.session_state.get('rag_query', ''),
        height=100,
        placeholder="Ask anything about the ingested documents...",
        key="query_input"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Query Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retrieval_strategy = st.selectbox(
                "Retrieval Strategy",
                options=['hybrid', 'vector', 'multi_query', 'compression'],
                format_func=lambda x: {
                    'hybrid': '🔀 Hybrid (Vector + Keyword)',
                    'vector': '🎯 Vector Similarity',
                    'multi_query': '🔄 Multi-Query Expansion',
                    'compression': '📦 Contextual Compression'
                }[x],
                help="Choose how to retrieve relevant documents"
            )
        
        with col2:
            num_sources = st.slider(
                "Number of Sources",
                min_value=1,
                max_value=20,
                value=5,
                help="How many sources to retrieve"
            )
        
        with col3:
            use_reranking = st.checkbox(
                "Use Re-ranking",
                value=True,
                help="Re-rank results for better precision"
            )
        
        # Metadata filters
        st.markdown("**Filter by Metadata:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_source_type = st.multiselect(
                "Source Type",
                options=['sec_filing', 'financial_report', 'research_report', 
                        'web_article', 'earnings_call'],
                default=None
            )
        
        with col2:
            filter_ticker = st.text_input("Ticker Symbol", placeholder="AAPL, MSFT, ...")
        
        with col3:
            filter_year = st.number_input(
                "Fiscal Year",
                min_value=2000,
                max_value=2025,
                value=None,
                step=1
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            filter_filing_type = st.multiselect(
                "Filing Type",
                options=['10-K', '10-Q', '8-K', 'DEF 14A'],
                default=None
            )
        
        with col2:
            filter_section = st.multiselect(
                "Document Section",
                options=['Business', 'Risk Factors', 'Management Discussion', 
                        'Financial Statements', 'Legal Proceedings'],
                default=None
            )
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_clicked = st.button(
            "🔍 Search Knowledge Base",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.rag_query = ''
            st.rerun()
    
    # Execute search
    if search_clicked and query:
        # Build filter dict
        filter_dict = {}
        
        if filter_source_type:
            filter_dict['source_type'] = filter_source_type
        if filter_ticker:
            filter_dict['ticker'] = filter_ticker.upper()
        if filter_year:
            filter_dict['fiscal_year'] = filter_year
        if filter_filing_type:
            filter_dict['filing_type'] = filter_filing_type
        if filter_section:
            filter_dict['section'] = filter_section
        
        # Execute query
        with st.spinner("🧠 Searching knowledge base..."):
            result = rag.query(
                question=query,
                retrieval_strategy=retrieval_strategy,
                filter_dict=filter_dict if filter_dict else None,
                num_sources=num_sources,
                use_reranking=use_reranking
            )
        
        if 'error' in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Display answer
            st.markdown("---")
            st.markdown("### 💡 Answer")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem; border-radius: 15px; color: white;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
                    <div style="color: white !important;">
        {result['answer']}
    </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Confidence", f"{result['confidence']*100:.0f}%")
            with col2:
                st.metric("Sources Used", result['num_sources_used'])
            with col3:
                st.metric("Strategy", result['retrieval_strategy'])
            with col4:
                quality_score = "High" if result['confidence'] > 0.7 else "Medium" if result['confidence'] > 0.4 else "Low"
                st.metric("Quality", quality_score)
            
            # Sources
            st.markdown("---")
            st.markdown("### 📚 Sources")
            
            for i, source in enumerate(result['sources'], 1):
                with st.expander(
                    f"Source {i} - Relevance: {source['relevance_score']:.3f}",
                    expanded=(i <= 2)
                ):
                    # Source metadata
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {source.get('source_type', 'N/A')}")
                        if source.get('ticker'):
                            st.write(f"**Ticker:** {source['ticker']}")
                        if source.get('company_name'):
                            st.write(f"**Company:** {source['company_name']}")
                    
                    with col2:
                        if source.get('filing_type'):
                            st.write(f"**Filing:** {source['filing_type']}")
                        if source.get('fiscal_year'):
                            st.write(f"**Year:** {source['fiscal_year']}")
                        if source.get('section'):
                            st.write(f"**Section:** {source['section']}")
                    
                    # Content
                    st.markdown("**Content:**")
                    st.write(source['text'])
                    
                    # URL if available
                    if source.get('url'):
                        st.markdown(f"[View Original Document]({source['url']})")
            
            # Save query to history
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            
            st.session_state.query_history.append({
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'confidence': result['confidence'],
                'num_sources': result['num_sources_used']
            })


def show_ingestion_interface(rag: AdvancedRAGSystem):
    """Data ingestion interface"""
    
    st.subheader("📥 Bulk Data Ingestion")
    
    # Ingestion type selector
    ingest_type = st.selectbox(
        "Ingestion Method",
        options=[
            "SEC Filings (Automatic)",
            "PDF Directory",
            "URL List",
            "Individual PDF Upload"
        ]
    )
    
    # ========================================================================
    # SEC FILINGS INGESTION
    # ========================================================================
    
    if ingest_type == "SEC Filings (Automatic)":
        st.markdown("### 📑 Ingest SEC Filings")
        
        st.info("""
        **Automatic SEC EDGAR Integration**
        
        This will fetch and process SEC filings (10-K, 10-Q, 8-K) for selected companies.
        Each filing is automatically sectioned and indexed for optimal retrieval.
        
        **What you'll get:**
        - Business descriptions
        - Risk factors
        - Financial statements
        - Management discussion & analysis
        - Legal proceedings
        """)
        
        # Get companies with tickers
        companies = st.session_state.db.get_all_companies()
        companies_with_tickers = [
            c for c in companies 
            if c.get('ticker_symbol') and c.get('is_public')
        ]
        
        if not companies_with_tickers:
            st.warning("⚠️ No companies with ticker symbols found.")
            st.info("Use the 'Discover Companies' feature to import public companies first.")
            return
        
        # Company selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Quick select options
            quick_select = st.selectbox(
                "Quick Select",
                options=['Custom', 'All Companies', 'Top 10 by Market Cap', 'Tech Companies', 'Healthcare Companies']
            )
            
            if quick_select == 'All Companies':
                selected_tickers = [c['ticker_symbol'] for c in companies_with_tickers]
            elif quick_select == 'Top 10 by Market Cap':
                sorted_companies = sorted(
                    companies_with_tickers,
                    key=lambda x: x.get('market_cap', 0),
                    reverse=True
                )
                selected_tickers = [c['ticker_symbol'] for c in sorted_companies[:10]]
            elif quick_select == 'Tech Companies':
                selected_tickers = [
                    c['ticker_symbol'] for c in companies_with_tickers
                    if c.get('industry') == 'Technology'
                ]
            elif quick_select == 'Healthcare Companies':
                selected_tickers = [
                    c['ticker_symbol'] for c in companies_with_tickers
                    if c.get('industry') == 'Healthcare'
                ]
            else:
                selected_tickers = st.multiselect(
                    "Select Companies",
                    options=[c['ticker_symbol'] for c in companies_with_tickers],
                    help="Choose companies to fetch SEC filings for"
                )
        
        with col2:
            filing_types = st.multiselect(
                "Filing Types",
                options=['10-K', '10-Q', '8-K', 'DEF 14A'],
                default=['10-K', '10-Q'],
                help="Types of SEC filings to fetch"
            )
            
            count_per_type = st.number_input(
                "Filings per Type",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of recent filings to fetch for each type"
            )
        
        # Estimate
        if selected_tickers:
            estimated_filings = len(selected_tickers) * len(filing_types) * count_per_type
            st.info(f"📊 Estimated filings to process: **{estimated_filings}**")
            
            estimated_time = estimated_filings * 0.5  # ~30 seconds per filing
            st.info(f"⏱️ Estimated time: **{estimated_time/60:.1f} minutes**")
        
        # Start ingestion
        if st.button("🚀 Start SEC Ingestion", type="primary", disabled=not selected_tickers):
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_container = st.empty()
            
            # Define progress callback
            def progress_callback(ticker, progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            # Run ingestion
            try:
                stats = asyncio.run(
                    rag.ingest_sec_filings_for_companies(
                        tickers=selected_tickers,
                        filing_types=filing_types,
                        count_per_type=count_per_type,
                        progress_callback=progress_callback
                    )
                )
                
                # Show results
                progress_bar.progress(1.0)
                status_text.success("✅ SEC filings ingestion complete!")
                
                # Display stats
                col1, col2, col3, col4 = stats_container.columns(4)
                
                with col1:
                    st.metric("Companies Processed", stats['tickers_processed'])
                with col2:
                    st.metric("Filings Downloaded", stats['filings_downloaded'])
                with col3:
                    st.metric("Documents Created", stats['documents_created'])
                with col4:
                    st.metric("Chunks Indexed", stats['chunks_indexed'])
                
                # Show errors if any
                if stats['errors']:
                    with st.expander("⚠️ Errors", expanded=False):
                        for error in stats['errors']:
                            st.error(f"{error['ticker']}: {error['error']}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # ========================================================================
    # PDF DIRECTORY INGESTION
    # ========================================================================
    
    elif ingest_type == "PDF Directory":
        st.markdown("### 📁 Ingest PDFs from Directory")
        
        st.info("""
        **Batch PDF Processing**
        
        Point to a directory containing PDF files (financial reports, research papers, etc.)
        and they will all be processed and indexed automatically.
        
        **Supported formats:**
        - Annual reports
        - Quarterly reports
        - Research papers
        - White papers
        - Case studies
        """)
        
        directory_path = st.text_input(
            "Directory Path",
            value="./data/pdfs",
            help="Absolute or relative path to directory containing PDFs"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            recursive = st.checkbox(
                "Search Subdirectories",
                value=True,
                help="Include PDFs in subdirectories"
            )
        
        with col2:
            source_type = st.selectbox(
                "Document Type",
                options=[
                    'financial_report',
                    'research_paper',
                    'case_study',
                    'whitepaper',
                    'earnings_call'
                ]
            )
        
        # Preview files
        dir_path = Path(directory_path)
        if dir_path.exists():
            if recursive:
                pdf_files = list(dir_path.rglob('*.pdf'))
            else:
                pdf_files = list(dir_path.glob('*.pdf'))
            
            st.success(f"✅ Found {len(pdf_files)} PDF files")
            
            if pdf_files and len(pdf_files) <= 10:
                with st.expander("📄 Preview Files"):
                    for pdf in pdf_files:
                        st.write(f"• {pdf.name}")
        else:
            st.error(f"❌ Directory not found: {directory_path}")
            pdf_files = []
        
        # Start ingestion
        if st.button("📥 Ingest PDFs", type="primary", disabled=not pdf_files):
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_container = st.empty()
            
            def progress_callback(filename, progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            try:
                stats = asyncio.run(
                    rag.ingest_pdfs_from_directory(
                        directory=dir_path,
                        source_type=source_type,
                        recursive=recursive,
                        progress_callback=progress_callback
                    )
                )
                
                progress_bar.progress(1.0)
                status_text.success("✅ PDF ingestion complete!")
                
                # Display stats
                col1, col2, col3, col4 = stats_container.columns(4)
                
                with col1:
                    st.metric("PDFs Found", stats['pdfs_found'])
                with col2:
                    st.metric("PDFs Processed", stats['pdfs_processed'])
                with col3:
                    st.metric("Documents Created", stats['documents_created'])
                with col4:
                    st.metric("Chunks Indexed", stats['chunks_indexed'])
                
                if stats['errors']:
                    with st.expander("⚠️ Errors"):
                        for error in stats['errors']:
                            st.error(f"{error['file']}: {error['error']}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
    
    # ========================================================================
    # URL LIST INGESTION
    # ========================================================================
    
    elif ingest_type == "URL List":
        st.markdown("### 🌐 Ingest from URLs")
        
        st.info("""
        **Web Content Ingestion**
        
        Scrape and index content from web pages. Great for:
        - News articles
        - Blog posts
        - Press releases
        - Company announcements
        """)
        
        urls_text = st.text_area(
            "URLs (one per line)",
            height=200,
            placeholder="https://example.com/article1\nhttps://example.com/article2"
        )
        
        # Parse URLs
        if urls_text:
            urls = []
            for line in urls_text.strip().split('\n'):
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append({'url': url})
            
            st.info(f"📊 {len(urls)} valid URLs found")
        else:
            urls = []
        
        if st.button("🌐 Ingest URLs", type="primary", disabled=not urls):
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_container = st.empty()
            
            def progress_callback(url, progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            try:
                stats = asyncio.run(
                    rag.ingest_urls(
                        urls=urls,
                        progress_callback=progress_callback
                    )
                )
                
                progress_bar.progress(1.0)
                status_text.success("✅ URL ingestion complete!")
                
                col1, col2, col3 = stats_container.columns(3)
                
                with col1:
                    st.metric("URLs Processed", stats['urls_processed'])
                with col2:
                    st.metric("Documents Created", stats['documents_created'])
                with col3:
                    st.metric("Chunks Indexed", stats['chunks_indexed'])
                
                if stats['errors']:
                    with st.expander("⚠️ Errors"):
                        for error in stats['errors']:
                            st.error(f"{error['url']}: {error['error']}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
    
    # ========================================================================
    # INDIVIDUAL PDF UPLOAD
    # ========================================================================
    
    elif ingest_type == "Individual PDF Upload":
        st.markdown("### 📤 Upload Individual PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a single PDF file to process"
        )
        
        if uploaded_file:
            # Save temporarily
            temp_dir = Path("./data/temp_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Metadata input
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("Company Name (optional)")
                ticker = st.text_input("Ticker Symbol (optional)")
            
            with col2:
                doc_type = st.selectbox(
                    "Document Type",
                    ['financial_report', 'research_paper', 'whitepaper']
                )
                fiscal_year = st.number_input(
                    "Fiscal Year (optional)",
                    min_value=2000,
                    max_value=2025,
                    value=None
                )
            
            if st.button("📥 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process single PDF
                        stats = asyncio.run(
                            rag.ingest_pdfs_from_directory(
                                directory=temp_dir,
                                source_type=doc_type,
                                recursive=False
                            )
                        )
                        
                        st.success("✅ PDF processed successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Documents Created", stats['documents_created'])
                        with col2:
                            st.metric("Chunks Indexed", stats['chunks_indexed'])
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")


def show_statistics_dashboard(rag: AdvancedRAGSystem):
    """Statistics and analytics dashboard"""
    
    st.subheader("📊 Knowledge Base Statistics")
    
    # Get statistics
    try:
        stats = rag.get_statistics()
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                f"{stats['collection_info'].get('points_count', 0):,}"
            )
        
        with col2:
            st.metric(
                "SEC Filings",
                f"{stats['ingestion_stats'].get('sec_filings', 0):,}"
            )
        
        with col3:
            st.metric(
                "PDFs",
                f"{stats['ingestion_stats'].get('pdfs', 0):,}"
            )
        
        with col4:
            st.metric(
                "URLs",
                f"{stats['ingestion_stats'].get('urls', 0):,}"
            )
        
        # Collection info
        st.markdown("---")
        st.markdown("### 🗄️ Collection Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Collection Name:** {stats['collection_info'].get('name', 'N/A')}")
            st.write(f"**Status:** {stats['collection_info'].get('status', 'N/A')}")
        
        with col2:
            st.write(f"**Vector Size:** {stats['collection_info'].get('vector_size', 0)}")
            st.write(f"**Distance Metric:** {stats['collection_info'].get('distance', 'N/A')}")
        
        with col3:
            last_ingestion = stats['ingestion_stats'].get('last_ingestion')
            if last_ingestion:
                last_date = datetime.fromisoformat(last_ingestion).strftime('%Y-%m-%d %H:%M')
                st.write(f"**Last Ingestion:** {last_date}")
        
        # Ingestion breakdown chart
        if any(stats['ingestion_stats'].values()):
            st.markdown("---")
            st.markdown("### 📈 Ingestion Breakdown")
            
            ingestion_data = {
                'Source': ['SEC Filings', 'PDFs', 'URLs'],
                'Documents': [
                    stats['ingestion_stats'].get('sec_filings', 0),
                    stats['ingestion_stats'].get('pdfs', 0),
                    stats['ingestion_stats'].get('urls', 0)
                ]
            }
            
            fig = px.pie(
                ingestion_data,
                values='Documents',
                names='Source',
                title="Documents by Source Type",
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Query history
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📜 Recent Queries")
            
            history_df = pd.DataFrame(st.session_state.query_history[-10:])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                history_df[['timestamp', 'query', 'confidence', 'num_sources']],
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error loading statistics: {e}")


def show_advanced_search(rag: AdvancedRAGSystem):
    """Advanced search with similarity exploration"""
    
    st.subheader("🔍 Advanced Search & Exploration")
    
    st.markdown("### 🔎 Find Similar Partnerships")
    
    st.info("""
    Use RAG to find similar historical partnerships based on:
    - Company profiles
    - Industry combinations
    - Partnership structures
    - Success patterns
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Company 1**")
        company1_id = show_company_selector(key_prefix="adv_search_c1")
    
    with col2:
        st.markdown("**Company 2**")
        exclude = [company1_id] if company1_id else None
        company2_id = show_company_selector(key_prefix="adv_search_c2", exclude_ids=exclude)
    
    num_results = st.slider("Number of Results", 1, 10, 5)
    
    if st.button("🔍 Find Similar Partnerships", disabled=not (company1_id and company2_id)):
        with st.spinner("Searching for similar partnerships..."):
            try:
                results = rag.find_similar_partnerships(
                    company1_id=company1_id,
                    company2_id=company2_id,
                    limit=num_results
                )
                
                if results:
                    st.success(f"✅ Found {len(results)} similar partnerships")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Match {i} - Relevance: {result.get('score', 0):.3f}"):
                            st.write(result['text'])
                            
                            metadata = result.get('metadata', {})
                            if metadata:
                                st.json(metadata)
                else:
                    st.info("No similar partnerships found. Try ingesting more data.")
            
            except Exception as e:
                st.error(f"Search failed: {e}")


def show_settings_panel(rag: AdvancedRAGSystem):
    """Settings and maintenance panel"""
    
    st.subheader("⚙️ Settings & Maintenance")
        # Vector Store Settings
    st.markdown("### 🗄️ Vector Store Configuration")
    
    current_type = st.session_state.get('vector_store_type', 'chroma')
    
    st.info(f"""
    **Current Vector Store:** {current_type.upper()}
    
    - **ChromaDB**: Best for most use cases, persistent storage, metadata filtering
    - **FAISS**: Fastest search, lower memory, good for < 1M documents
    
    Change requires restart and will create a new index.
    """)
    
    new_type = st.selectbox(
        "Select Vector Store",
        options=['chroma', 'faiss'],
        index=['chroma', 'faiss'].index(current_type)
    )
    
    if new_type != current_type:
        if st.button("Apply and Restart"):
            st.session_state.vector_store_type = new_type
            if 'advanced_rag' in st.session_state:
                del st.session_state.advanced_rag
            st.success(f"Switched to {new_type.upper()}. Reloading...")
            st.rerun()
    # Database maintenance
    st.markdown("### 🗄️ Database Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Statistics"):
            st.rerun()
    
    with col2:
        if st.button("💾 Export Knowledge Base"):
            try:
                output_file = Path(f"./data/exports/kb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with st.spinner("Exporting..."):
                    rag.export_knowledge_base(output_file)
                
                st.success(f"✅ Exported to: {output_file}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    # Dangerous operations
    st.markdown("---")
    st.markdown("### ⚠️ Dangerous Operations")
    
    with st.expander("🗑️ Delete Documents", expanded=False):
        st.warning("**Warning:** This will permanently delete documents from the vector store.")
        
        delete_by = st.selectbox(
            "Delete by",
            options=['Ticker', 'Source Type', 'Date Range', 'All Documents']
        )
        
        if delete_by == 'Ticker':
            ticker_to_delete = st.text_input("Ticker Symbol")
            
            if st.button("Delete", type="secondary"):
                if ticker_to_delete and st.checkbox("I understand this is permanent"):
                    count = rag.delete_documents_by_filter({'ticker': ticker_to_delete.upper()})
                    st.success(f"Deleted {count} documents")
        
        elif delete_by == 'All Documents':
            st.error("**DANGER:** This will delete ALL documents!")
            
            if st.text_input("Type 'DELETE ALL' to confirm") == "DELETE ALL":
                if st.button("Delete All Documents", type="secondary"):
                    # This would require implementing a delete_all method
                    st.warning("Not implemented for safety")


def show_company_selector(key_prefix: str, exclude_ids: Optional[list] = None):
    """Company selector helper"""
    companies = st.session_state.db.get_all_companies()
    
    if exclude_ids:
        companies = [c for c in companies if c['id'] not in exclude_ids]
    
    if not companies:
        st.warning("No companies available")
        return None
    
    company_options = {
        f"{c.get('name', 'Unknown')} ({c.get('ticker_symbol', 'N/A')})": c['id']
        for c in companies
    }
    
    selected = st.selectbox(
        "Select Company",
        options=[""] + list(company_options.keys()),
        key=f"{key_prefix}_selector"
    )
    
    return company_options.get(selected) if selected else None
