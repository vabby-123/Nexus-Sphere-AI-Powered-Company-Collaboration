"""
Check if ChromaDB actually has data
"""

import os
from pathlib import Path

# Check if the database directory exists
chroma_dir = Path("./chroma_db")

if chroma_dir.exists():
    print(f"✅ ChromaDB directory exists: {chroma_dir}")
    
    # List all files
    files = list(chroma_dir.rglob("*"))
    print(f"\n📁 Files in ChromaDB directory ({len(files)} total):")
    for f in files[:10]:  # Show first 10
        print(f"  - {f}")
    
    # Try to load and inspect the collection
    try:
        from chromadb import Client
        from chromadb.config import Settings
        
        client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(chroma_dir)
        ))
        
        # List collections
        collections = client.list_collections()
        print(f"\n📚 Collections found: {len(collections)}")
        
        for coll in collections:
            print(f"\n  Collection: {coll.name}")
            count = coll.count()
            print(f"  Documents: {count}")
            
            if count > 0:
                # Get a sample
                sample = coll.peek(limit=3)
                print(f"  Sample IDs: {sample['ids'][:3]}")
    
    except Exception as e:
        print(f"\n⚠️ Could not inspect ChromaDB directly: {e}")
        print("This is OK - trying alternative method...")
        
        # Alternative: Use our vector store
        try:
            from rag_system.vector_store_free import ChromaVectorStore
            
            store = ChromaVectorStore(
                collection_name="nexus_sphere_knowledge",
                persist_directory="./chroma_db"
            )
            
            info = store.get_collection_info()
            print(f"\n✅ Collection info from vector store:")
            print(f"  Points: {info.get('points_count', 0)}")
            
            # Try to get some points
            points, _ = store.scroll_points(limit=5)
            print(f"\n📄 Sample documents: {len(points)}")
            for i, point in enumerate(points[:3], 1):
                text = point['payload'].get('text', '')[:100]
                print(f"  {i}. {text}...")
                
        except Exception as e2:
            print(f"\n❌ Alternative method failed: {e2}")

else:
    print(f"❌ ChromaDB directory NOT found: {chroma_dir}")
    print("\n💡 Data was not saved. Possible reasons:")
    print("  1. Ingestion failed silently")
    print("  2. Wrong directory path")
    print("  3. Permissions issue")