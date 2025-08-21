#!/usr/bin/env python3
"""
HRM Matrix Integration Runner
Test the full optimization pipeline
"""

import asyncio
from hrm_matrix_integration import HRMMatrixIntegrator
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

async def main():
    print("🚀 Starting HRM Matrix Integration Test...")
    
    # Create a minimal config for testing
    cfg = {
        "batch_size": 1, 
        "seq_len": 8, 
        "puzzle_emb_ndim": 64,
        "num_puzzle_identifiers": 10, 
        "vocab_size": 1000,
        "H_cycles": 1, 
        "L_cycles": 1, 
        "H_layers": 1, 
        "L_layers": 1,
        "hidden_size": 128, 
        "expansion": 4.0, 
        "num_heads": 4,
        "pos_encodings": "rope", 
        "halt_max_steps": 4, 
        "halt_exploration_prob": 0.1
    }
    
    try:
        print("📦 Creating HRM model...")
        model = HierarchicalReasoningModel_ACTV1(cfg)
        print(f"✅ HRM model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print("🔧 Initializing Matrix Integrator...")
        integrator = HRMMatrixIntegrator()
        
        print("🚀 Starting integration...")
        await integrator.start()
        
        print("🧪 Running entropy analysis...")
        entropy_results = await integrator.analyze_hrm_entropy(model)
        print(f"✅ Entropy analysis completed: {entropy_results}")
        
        print("🚀 Running full optimization...")
        out = await integrator.optimize_hrm_weights(
            model, 
            optimization_method="sparsity", 
            poly_degree=3, 
            run_id="hrm_opt_demo"
        )
        
        if out:
            print("✅ Optimization completed successfully!")
            print(f"📊 Results: {out.get('export', out)}")
        else:
            print("⚠️  Optimization returned no results")
        
        print("🛑 Stopping integration...")
        await integrator.stop()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
