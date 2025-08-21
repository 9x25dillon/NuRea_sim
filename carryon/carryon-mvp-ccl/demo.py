#!/usr/bin/env python3
"""
CarryOn MVP Demo Script
Demonstrates the system's capabilities
"""

import sys
import os
import json
from datetime import datetime

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

def demo_memory_system():
    """Demonstrate memory creation and management"""
    print("🧠 Memory System Demo")
    print("-" * 30)
    
    try:
        from app.models.memory_event import MemoryEvent
        
        # Create some sample memories
        memories = [
            MemoryEvent(data="The user prefers concise, technical explanations", subject="user_preferences"),
            MemoryEvent(data="Python is the primary programming language used", subject="technology"),
            MemoryEvent(data="The user is working on an AI assistant project", subject="project_context"),
            MemoryEvent(data="Previous discussions covered vector embeddings and FAISS", subject="technical_topics")
        ]
        
        print(f"✅ Created {len(memories)} sample memories")
        for i, mem in enumerate(memories, 1):
            print(f"   {i}. {mem.data[:50]}... (recency: {mem.recency_score:.3f})")
        
        return memories
        
    except Exception as e:
        print(f"❌ Memory demo failed: {e}")
        return []

def demo_vector_search():
    """Demonstrate vector search capabilities"""
    print("\n🔍 Vector Search Demo")
    print("-" * 30)
    
    try:
        from app.retrieval.vector_index import rebuild_index, search
        
        # Get memories from previous demo
        memories = demo_memory_system()
        if not memories:
            return
        
        # Rebuild index
        print("🔄 Rebuilding vector index...")
        result = rebuild_index(memories)
        print(f"✅ Index built: {result}")
        
        # Test search
        query = "AI assistant programming"
        print(f"\n🔎 Searching for: '{query}'")
        results = search(query, top_n=3)
        
        if results:
            print("✅ Search results:")
            for i, (mem_id, score) in enumerate(results, 1):
                print(f"   {i}. ID: {mem_id[:8]}... (score: {score:.3f})")
        else:
            print("⚠️  No search results")
            
    except Exception as e:
        print(f"❌ Vector search demo failed: {e}")

def demo_tone_alignment():
    """Demonstrate tone alignment functionality"""
    print("\n🎭 Tone Alignment Demo")
    print("-" * 30)
    
    try:
        from app.tone.alignment import evaluate_tone
        
        # Define a persona voice
        persona_voice = {
            "tone": ["professional", "helpful", "concise"],
            "style_rules": [
                "avoid purple prose",
                "use concrete examples", 
                "match user energy",
                "be technically accurate"
            ]
        }
        
        print("🎯 Persona Voice:")
        print(f"   Tone: {', '.join(persona_voice['tone'])}")
        print(f"   Rules: {', '.join(persona_voice['style_rules'])}")
        
        # Test different candidate texts
        test_texts = [
            "This is a wonderfully amazing and incredibly fantastic response that demonstrates excellent communication skills!",
            "Here's a concise solution: use the vector index for semantic search. It supports both FAISS and TF-IDF fallback.",
            "I think maybe this could possibly work, but I'm not entirely sure about the implementation details."
        ]
        
        print("\n📝 Testing candidate texts:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n   {i}. Text: '{text[:60]}...'")
            result = evaluate_tone(persona_voice, text)
            print(f"      Score: {result['score']}/100")
            print(f"      Metrics: {result['metrics']}")
            if result['suggestions']:
                print(f"      Suggestions: {', '.join(result['suggestions'])}")
                
    except Exception as e:
        print(f"❌ Tone alignment demo failed: {e}")

def demo_ranking():
    """Demonstrate memory ranking system"""
    print("\n🏆 Memory Ranking Demo")
    print("-" * 30)
    
    try:
        from app.retrieval.ranker import rank_memories
        
        # Get memories
        memories = demo_memory_system()
        if not memories:
            return
        
        # Test ranking
        query = "AI programming assistance"
        print(f"🔍 Ranking memories for query: '{query}'")
        
        ranked = rank_memories(query, memories, k=3, entropy=0.2)
        
        print("✅ Ranking results:")
        for i, mem in enumerate(ranked, 1):
            print(f"   {i}. {mem.data[:50]}...")
            print(f"      Subject: {mem.subject}")
            print(f"      Recency: {mem.recency_score:.3f}")
            
    except Exception as e:
        print(f"❌ Ranking demo failed: {e}")

def main():
    """Run the complete demo"""
    print("🚀 CarryOn MVP System Demo")
    print("=" * 50)
    print("This demo showcases the system's core capabilities:\n")
    
    demos = [
        demo_memory_system,
        demo_vector_search,
        demo_tone_alignment,
        demo_ranking
    ]
    
    for demo in demos:
        try:
            demo()
            print()  # Add spacing between demos
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Start the server: cd server && python start.py")
    print("   2. Open the Electron app: cd apps/desktop-electron && npm start")
    print("   3. Test the API endpoints with the provided examples")

if __name__ == "__main__":
    main() 