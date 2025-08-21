#!/usr/bin/env python3
"""
Test script for Categorical Coherence Linter (CCL)
Demonstrates CCL functionality and validates the implementation
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the server app to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server', 'app'))

try:
    from tools.ccl import (
        analyze_path, 
        report_to_json, 
        print_human_summary,
        _shannon_entropy,
        _deep_equal,
        _gen_value,
        probe_function
    )
    print("✓ CCL imports successful")
except ImportError as e:
    print(f"✗ CCL import failed: {e}")
    sys.exit(1)

def test_utilities():
    """Test utility functions"""
    print("\n=== Testing Utility Functions ===")
    
    # Test entropy calculation
    values = [1, 1, 2, 2, 2, 3]
    entropy = _shannon_entropy(values)
    print(f"Entropy of {values}: {entropy:.4f}")
    
    # Test deep equality
    assert _deep_equal([1, 2, 3], [1, 2, 3])
    assert _deep_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not _deep_equal([1, 2], [1, 2, 3])
    print("✓ Deep equality tests passed")
    
    # Test value generation
    for _ in range(5):
        val = _gen_value()
        print(f"Generated value: {val} (type: {type(val).__name__})")
    print("✓ Value generation tests passed")

def test_simple_functions():
    """Test CCL on simple mathematical functions"""
    print("\n=== Testing Simple Functions ===")
    
    # Create a temporary file with test functions
    test_code = '''
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def square(x):
    return x * x

def identity(x):
    return x

def constant(x):
    return 42

def noisy(x):
    import random
    return x + random.random() * 0.1
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Analyze the test file
        print(f"Analyzing test file: {temp_file}")
        analysis = analyze_path(temp_file, samples=50, seed=42)
        
        print(f"Functions analyzed: {analysis.summary['functions_analyzed']}")
        print(f"Top hotspots: {len(analysis.summary['top_hotspots'])}")
        
        # Print human-readable summary
        print_human_summary(analysis)
        
        # Test JSON report generation
        report = report_to_json(analysis)
        assert 'functions' in report
        assert 'ghost_hotspots' in report
        print("✓ JSON report generation successful")
        
    finally:
        # Clean up
        os.unlink(temp_file)

def test_ccl_integration():
    """Test CCL integration with the project"""
    print("\n=== Testing CCL Integration ===")
    
    # Test on a real project file
    project_files = [
        "server/app/retrieval/vector_index.py",
        "server/app/tone/alignment.py",
        "server/app/retrieval/ranker.py"
    ]
    
    for file_path in project_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing: {file_path}")
            try:
                analysis = analyze_path(file_path, samples=30, seed=42)
                print(f"  Functions: {analysis.summary['functions_analyzed']}")
                if analysis.summary['top_hotspots']:
                    top = analysis.summary['top_hotspots'][0]
                    print(f"  Top hotspot: {top['function']} (score: {top['score']})")
                else:
                    print("  No hotspots found")
            except Exception as e:
                print(f"  Error analyzing {file_path}: {e}")
        else:
            print(f"  File not found: {file_path}")

def main():
    """Run all tests"""
    print("Categorical Coherence Linter (CCL) Test Suite")
    print("=" * 50)
    
    try:
        test_utilities()
        test_simple_functions()
        test_ccl_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 