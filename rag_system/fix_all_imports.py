"""
Automatically fix all import statements in rag_system
"""

from pathlib import Path
import re

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    content = file_path.read_text(encoding='utf-8')
    original = content
    
    # Patterns to fix (add dot for relative imports)
    fixes = [
        (r'^from data_fetchers import', 'from .data_fetchers import'),
        (r'^from document_processors import', 'from .document_processors import'),
        (r'^from vector_store import', 'from .vector_store import'),
        (r'^from vector_store_free import', 'from .vector_store_free import'),
        (r'^from vector_Store import', 'from .vector_store import'),  # Fix capitalization too
        (r'^from advanced_rag import', 'from .advanced_rag import'),
        (r'^from retrieval_strategies import', 'from .retrieval_strategies import'),
        (r'^from rag_ui import', 'from .rag_ui import'),
        (r'from rag_system\.document_processors import', 'from .document_processors import'),  # Remove redundant rag_system.
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Also add missing imports
    if 'advanced_rag.py' in str(file_path):
        # Make sure 'os' is imported
        if 'import os' not in content and 'os.getenv' in content:
            # Find the imports section and add os
            content = content.replace(
                'from pathlib import Path',
                'from pathlib import Path\nimport os'
            )
    
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ Fixed: {file_path.name}")
        return True
    else:
        print(f"⏭️  No changes: {file_path.name}")
        return False

# Fix all Python files in rag_system
rag_dir = Path('rag_system')
fixed_count = 0

python_files = [
    'advanced_rag.py',
    'data_fetchers.py',
    'document_processors.py',
    'rag_ui.py',
    'retrieval_strategies.py',
    'vector_store.py',
    'vector_store_free.py'
]

for filename in python_files:
    file_path = rag_dir / filename
    if file_path.exists():
        if fix_imports_in_file(file_path):
            fixed_count += 1

print(f"\n✅ Fixed {fixed_count} files")
print("\n🎯 Now run: python test_rag_imports.py")