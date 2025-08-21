#!/usr/bin/env python3
"""
Simple test script for HRM Matrix Integration
Run this to verify everything is working
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test Matrix Orchestrator import
        from matrix_orchestrator import MatrixOrchestrator, MatrixChunk
        print("✅ Matrix Orchestrator imported successfully")
        
        # Test HRM model import
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        print("✅ HRM model imported successfully")
        
        # Test integration import
        from matrix_integration import HRMMatrixIntegrator
        print("✅ Matrix Integration imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_matrix_orchestrator():
    """Test that Matrix Orchestrator can start and stop."""
    print("\n🧪 Testing Matrix Orchestrator...")
    
    try:
        from matrix_orchestrator import MatrixOrchestrator
        
        orchestrator = MatrixOrchestrator()
        print("✅ Matrix Orchestrator created")
        
        # Test start
        result = await orchestrator.start()
        print(f"✅ Matrix Orchestrator started: {result}")
        
        # Test health check
        health = await orchestrator.health_check()
        print(f"✅ Health check: {health}")
        
        # Test stop
        result = await orchestrator.stop()
        print(f"✅ Matrix Orchestrator stopped: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix Orchestrator test failed: {e}")
        return False

async def test_hrm_model_creation():
    """Test that we can create an HRM model."""
    print("\n🧪 Testing HRM model creation...")
    
    try:
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        
        # Create a minimal config for testing
        config = {
            "batch_size": 1,
            "seq_len": 8,
            "puzzle_emb_ndim": 32,
            "num_puzzle_identifiers": 5,
            "vocab_size": 100,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 64,
            "expansion": 2.0,
            "num_heads": 2,
            "pos_encodings": "rope",
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.1
        }
        
        model = HierarchicalReasoningModel_ACTV1(config)
        print("✅ HRM model created successfully")
        
        # Test that it has parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model has {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ HRM model test failed: {e}")
        return False

async def test_matrix_integration():
    """Test the full integration."""
    print("\n🧪 Testing Matrix Integration...")
    
    try:
        from matrix_integration import HRMMatrixIntegrator
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        
        # Create a minimal config
        config = {
            "batch_size": 1,
            "seq_len": 8,
            "puzzle_emb_ndim": 32,
            "num_puzzle_identifiers": 5,
            "vocab_size": 100,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 64,
            "expansion": 2.0,
            "num_heads": 2,
            "pos_encodings": "rope",
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.1
        }
        
        model = HierarchicalReasoningModel_ACTV1(config)
        integrator = HRMMatrixIntegrator()
        
        print("✅ Integration objects created")
        
        # Test start
        await integrator.start()
        print("✅ Integration started")
        
        # Test entropy analysis
        entropy_results = await integrator.analyze_hrm_entropy(model)
        print(f"✅ Entropy analysis completed: {entropy_results}")
        
        # Test stop
        await integrator.stop()
        print("✅ Integration stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting HRM Matrix Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_matrix_orchestrator,
        test_hrm_model_creation,
        test_matrix_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your integration is working!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
