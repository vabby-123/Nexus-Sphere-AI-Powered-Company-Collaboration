"""
Fix query result dictionary to always include all required keys
"""

from pathlib import Path
import re

file_path = Path('rag_system/advanced_rag.py')
content = file_path.read_text(encoding='utf-8')

# Fix the main return statement
content = re.sub(
    r"return \{\s*'answer': answer,\s*'sources': self\._format_sources\(results\),\s*'confidence': confidence,\s*'retrieval_strategy': retrieval_strategy\s*\}",
    """return {
            'answer': answer,
            'sources': self._format_sources(results),
            'confidence': confidence,
            'retrieval_strategy': retrieval_strategy,
            'num_sources_used': len(results)
        }""",
    content,
    flags=re.DOTALL
)

# Fix the no results case
content = re.sub(
    r"return \{\s*'answer': 'No relevant information found in the knowledge base\.',\s*'sources': \[\],\s*'confidence': 0\.0,\s*'retrieval_strategy': retrieval_strategy\s*\}",
    """return {
                'answer': 'No relevant information found in the knowledge base.',
                'sources': [],
                'confidence': 0.0,
                'retrieval_strategy': retrieval_strategy,
                'num_sources_used': 0
            }""",
    content,
    flags=re.DOTALL
)

# Fix the error case
content = re.sub(
    r"return \{\s*'error': str\(e\),\s*'answer': 'An error occurred while processing your query\.',\s*'sources': \[\]\s*\}",
    """return {
                'error': str(e),
                'answer': 'An error occurred while processing your query.',
                'sources': [],
                'num_sources_used': 0,
                'confidence': 0.0,
                'retrieval_strategy': retrieval_strategy
            }""",
    content,
    flags=re.DOTALL
)

file_path.write_text(content, encoding='utf-8')
print("✅ Fixed query result dictionary in advanced_rag.py")