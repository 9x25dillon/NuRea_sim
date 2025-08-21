from typing import List, Dict
from collections import Counter

def degree(subjects: List[str]) -> Dict[str, float]:
    """Calculate degree centrality for subjects (simple frequency-based)"""
    if not subjects:
        return {}
    
    # Count occurrences of each subject
    subject_counts = Counter(subjects)
    max_count = max(subject_counts.values()) if subject_counts else 1
    
    # Normalize to 0-1 range
    return {subject: count / max_count for subject, count in subject_counts.items()} 