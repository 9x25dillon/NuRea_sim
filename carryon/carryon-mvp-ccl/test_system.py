#!/usr/bin/env python3
"""
Test script for CarryOn MVP system components
"""

import sys
import os

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from app.config.settings import settings
        print("✅ Settings imported")
    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    try:
        from app.models.memory_event import MemoryEvent
        print("✅ MemoryEvent model imported")
    except Exception as e:
        print(f"❌ MemoryEvent import failed: {e}")
        return False
    
    try:
        from app.retrieval.vector_index import search, rebuild_index
        print("✅ Vector index imported")
    except Exception as e:
        print(f"❌ Vector index import failed: {e}")
        return False
    
    try:
        from app.tone.alignment import evaluate_tone
        print("✅ Tone alignment imported")
    except Exception as e:
        print(f"❌ Tone alignment import failed: {e}")
        return False
    
    return True

def test_models():
    """Test model creation"""
    print("\n🧪 Testing models...")
    
    try:
        from app.models.memory_event import MemoryEvent
        from datetime import datetime
        
        # Create a test memory
        memory = MemoryEvent(
            data="This is a test memory for the CarryOn MVP system",
            subject="testing"
        )
        
        print(f"✅ Memory created: {memory.event_id}")
        print(f"   Recency score: {memory.recency_score:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_tone_alignment():
    """Test tone alignment functionality"""
    print("\n🧪 Testing tone alignment...")
    
    try:
        from app.tone.alignment import evaluate_tone
        
        # Test persona voice
        persona_voice = {
            "tone": ["professional", "friendly"],
            "style_rules": ["avoid purple prose", "use concrete examples"]
        }
        
        # Test candidate text
        candidate_text = "This is a wonderfully amazing and incredibly fantastic response that demonstrates excellent communication skills!"
        
        result = evaluate_tone(persona_voice, candidate_text)
        
        print(f"✅ Tone evaluation completed")
        print(f"   Score: {result['score']}/100")
        print(f"   Metrics: {result['metrics']}")
        print(f"   Suggestions: {result['suggestions']}")
        
        return True
    except Exception as e:
        print(f"❌ Tone alignment test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CarryOn MVP System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_models,
        test_tone_alignment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 