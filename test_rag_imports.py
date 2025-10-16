import sys

print("="*60)
print("Testing RAG System Imports")
print("="*60)

# Test individual imports
imports_to_test = [
    ("rag_system", "RAG system package"),
    ("rag_system.advanced_rag", "Main RAG orchestrator"),
    ("rag_system.vector_store_free", "Vector store (free)"),
    ("rag_system.data_fetchers", "Data fetchers"),
    ("rag_system.document_processors", "Document processors"),
    ("rag_system.rag_ui", "RAG UI"),
]

failed_imports = []

for module_name, description in imports_to_test:
    try:
        __import__(module_name)
        print(f"✅ {description}: OK")
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        failed_imports.append((module_name, str(e)))

print("\n" + "="*60)

if failed_imports:
    print("⚠️ Failed Imports:")
    for module, error in failed_imports:
        print(f"\n{module}:")
        print(f"  {error}")
else:
    print("✅ All imports successful!")

print("="*60)