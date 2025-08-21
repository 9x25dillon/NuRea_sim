"""
Matrix Orchestrator Integration for HRM Project

This module integrates the Enhanced Matrix Orchestrator with the HRM
(Hierarchical Reasoning Model) project for advanced mathematical optimization.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Import the matrix orchestrator
try:
    from matrix_orchestrator import (
        MatrixOrchestrator, MatrixChunk, PolySpec, OptimizeRequest,
        AdjacencyPayload, orchestrate, RunPlan
    )
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    logging.warning("Matrix Orchestrator not available - using fallback methods")

# Import HRM model
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

# Add time import for run_id generation
import time

log = logging.getLogger(__name__)


class HRMMatrixIntegrator:
    """
    Integrates Matrix Orchestrator with HRM for advanced optimization.
    """
    
    def __init__(self, matrix_config: Optional[Dict[str, Any]] = None):
        self.matrix_config = matrix_config or {}
        self.matrix_available = MATRIX_AVAILABLE
        
        if self.matrix_available:
            try:
                self.orchestrator = MatrixOrchestrator()
                log.info("✅ Matrix Orchestrator initialized")
            except Exception as e:
                log.warning(f"⚠️  Matrix Orchestrator failed to initialize: {e}")
                self.matrix_available = False
        else:
            log.info("ℹ️  Using fallback optimization methods")
    
    async def start(self) -> bool:
        """Start the matrix orchestrator."""
        if self.matrix_available:
            try:
                await self.orchestrator.start()
                return True
            except Exception as e:
                log.error(f"❌ Failed to start Matrix Orchestrator: {e}")
                self.matrix_available = False
                return False
        return True
    
    async def stop(self) -> bool:
        """Stop the matrix orchestrator."""
        if self.matrix_available:
            try:
                await self.orchestrator.stop()
                return True
            except Exception as e:
                log.error(f"❌ Failed to stop Matrix Orchestrator: {e}")
                return False
        return True
    
    def extract_matrix_from_hrm(self, hrm_model: HierarchicalReasoningModel_ACTV1) -> List[List[float]]:
        """
        Extract weight matrices from HRM model for optimization.
        
        Args:
            hrm_model: The HRM model instance
            
        Returns:
            List of weight matrices as 2D lists
        """
        matrices = []
        
        try:
            # Extract weights from different layers
            for name, module in hrm_model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_matrix = module.weight.detach().cpu().numpy()
                    if weight_matrix.ndim == 2:  # Only 2D matrices
                        matrices.append(weight_matrix.tolist())
                        log.debug(f"Extracted matrix from {name}: {weight_matrix.shape}")
            
            log.info(f"✅ Extracted {len(matrices)} weight matrices from HRM model")
            return matrices
            
        except Exception as e:
            log.error(f"❌ Failed to extract matrices from HRM model: {e}")
            return []
    
    def create_matrix_chunks(self, matrices: List[List[List[float]]], chunk_size: int = 1000) -> List[MatrixChunk]:
        """
        Create MatrixChunk objects from extracted matrices.
        
        Args:
            matrices: List of weight matrices
            chunk_size: Maximum size for each chunk
            
        Returns:
            List of MatrixChunk objects
        """
        if not self.matrix_available:
            return []
        
        chunks = []
        chunk_id = 0
        
        for matrix in matrices:
            # Guard against empty matrices
            if not matrix or not matrix[0]:
                continue
                
            H, W = len(matrix), len(matrix[0])
            
            # Split large matrices into chunks
            if H > chunk_size or W > chunk_size:
                # Create sub-matrices
                for i in range(0, H, chunk_size):
                    for j in range(0, W, chunk_size):
                        sub_matrix = [
                            row[j:j+chunk_size] 
                            for row in matrix[i:i+chunk_size]
                        ]
                        chunk = MatrixChunk(
                            id=f"hrm_chunk_{chunk_id}",
                            data=sub_matrix,
                            meta={"source": "hrm_weights", "i": i, "j": j}
                        )
                        chunks.append(chunk)
                        chunk_id += 1
            else:
                chunk = MatrixChunk(
                    id=f"hrm_chunk_{chunk_id}",
                    data=matrix,
                    meta={"source": "hrm_weights", "full_matrix": True}
                )
                chunks.append(chunk)
                chunk_id += 1
        
        log.info(f"✅ Created {len(chunks)} matrix chunks")
        return chunks
    
    async def optimize_hrm_weights(
        self,
        hrm_model: HierarchicalReasoningModel_ACTV1,
        optimization_method: str = "sparsity",
        poly_degree: int = 3,
        run_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Optimize HRM model weights using Matrix Orchestrator.
        
        Args:
            hrm_model: The HRM model to optimize
            optimization_method: Optimization method ("sparsity", "rank", "structure")
            poly_degree: Polynomial degree for Chebyshev projection
            run_id: Optional run ID for tracking
            
        Returns:
            Optimization results or None if failed
        """
        if not self.matrix_available:
            log.warning("⚠️  Matrix Orchestrator not available - skipping optimization")
            return None
        
        try:
            # Extract matrices from HRM model
            matrices = self.extract_matrix_from_hrm(hrm_model)
            if not matrices:
                log.warning("⚠️  No matrices extracted from HRM model")
                return None
            
            # Create matrix chunks
            chunks = self.create_matrix_chunks(matrices)
            if not chunks:
                log.warning("⚠️  No matrix chunks created")
                return None
            
            # Generate run ID if not provided
            if run_id is None:
                import time
                run_id = f"hrm_optimization_{int(time.time())}"
            
            # Create optimization plan
            plan = RunPlan(
                run_id=run_id,
                chunks=chunks,
                poly=PolySpec(degree=poly_degree, basis="chebyshev"),
                optimize=OptimizeRequest(
                    matrix=matrices[0],  # Use first matrix as template
                    method=optimization_method,
                    params={"max_iterations": 100, "tolerance": 1e-6}
                ),
                export_path="hrm_optimization_results.json"
            )
            
            # Run optimization
            log.info(f"🚀 Starting HRM weight optimization with {len(chunks)} chunks")
            results = await orchestrate(plan)
            
            log.info(f"✅ HRM optimization completed successfully")
            return results
            
        except Exception as e:
            log.error(f"❌ HRM optimization failed: {e}")
            return None
    
    async def analyze_hrm_entropy(self, hrm_model: HierarchicalReasoningModel_ACTV1) -> Dict[str, Any]:
        """
        Analyze entropy of HRM model weights.
        
        Args:
            hrm_model: The HRM model to analyze
            
        Returns:
            Entropy analysis results
        """
        if not self.matrix_available:
            return self._fallback_entropy_analysis(hrm_model)
        
        try:
            matrices = self.extract_matrix_from_hrm(hrm_model)
            # Sample a few matrices to keep it light and fast
            matrices = matrices[:8]  # Limit to first 8 matrices
            chunks = self.create_matrix_chunks(matrices, chunk_size=512)  # Smaller chunks for speed
            
            # Create a simple plan for entropy analysis
            plan = RunPlan(
                run_id=f"hrm_entropy_{int(time.time())}",
                chunks=chunks,
                poly=PolySpec(degree=1, basis="chebyshev"),
                optimize=OptimizeRequest(
                    matrix=[[1.0]],  # Dummy matrix
                    method="sparsity"
                ),
                export_path="hrm_entropy_results.json"
            )
            
            # Run only entropy analysis
            results = await orchestrate(plan)
            
            return {
                "entropy_map": results.get("entropy_map", []),
                "hot_chunks": len([r for r in results.get("entropy_map", []) 
                                 if r.get("shannon", 0.0) > 0.85]),
                "matrix_count": len(matrices),
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            log.error(f"❌ HRM entropy analysis failed: {e}")
            return self._fallback_entropy_analysis(hrm_model)
    
    def _fallback_entropy_analysis(self, hrm_model: HierarchicalReasoningModel_ACTV1) -> Dict[str, Any]:
        """Fallback entropy analysis when Matrix Orchestrator is unavailable."""
        try:
            matrices = self.extract_matrix_from_hrm(hrm_model)
            
            entropy_results = []
            for i, matrix in enumerate(matrices):
                arr = np.array(matrix)
                if arr.size == 0:
                    continue
                
                # Simple entropy calculation
                hist, _ = np.histogram(arr, bins=64, density=True)
                p = hist[hist > 0]
                shannon = float(-(p * np.log2(p)).sum()) if p.size else 0.0
                
                entropy_results.append({
                    "chunk_id": f"fallback_chunk_{i}",
                    "shannon": shannon,
                    "spectral": 0.0,  # Placeholder
                    "compression_ratio": 1.0  # Placeholder
                })
            
            return {
                "entropy_map": entropy_results,
                "hot_chunks": len([r for r in entropy_results if r.get("shannon", 0) > 0.85]),
                "matrix_count": len(matrices),
                "chunk_count": len(entropy_results),
                "fallback": True
            }
            
        except Exception as e:
            log.error(f"❌ Fallback entropy analysis failed: {e}")
            return {"error": str(e), "fallback": True}


# Convenience function for quick integration
async def integrate_matrix_with_hrm(
    hrm_model: HierarchicalReasoningModel_ACTV1,
    optimization_method: str = "sparsity",
    poly_degree: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Quick integration function for HRM + Matrix Orchestrator.
    
    Args:
        hrm_model: The HRM model to optimize
        optimization_method: Optimization method
        poly_degree: Polynomial degree
        
    Returns:
        Integration results or None
    """
    integrator = HRMMatrixIntegrator()
    
    try:
        await integrator.start()
        results = await integrator.optimize_hrm_weights(
            hrm_model, optimization_method, poly_degree
        )
        return results
    finally:
        await integrator.stop()


if __name__ == "__main__":
    # Test the integration
    async def test():
        print("🧪 Testing HRM Matrix Integration...")
        
        # Create a dummy HRM model for testing
        config_dict = {
            "batch_size": 1,
            "seq_len": 10,
            "puzzle_emb_ndim": 64,
            "num_puzzle_identifiers": 10,
            "vocab_size": 1000,
            "H_cycles": 3,
            "L_cycles": 2,
            "H_layers": 2,
            "L_layers": 2,
            "hidden_size": 128,
            "expansion": 4.0,
            "num_heads": 8,
            "pos_encodings": "rope",
            "halt_max_steps": 10,
            "halt_exploration_prob": 0.1
        }
        
        try:
            hrm_model = HierarchicalReasoningModel_ACTV1(config_dict)
            print("✅ HRM model created successfully")
            
            integrator = HRMMatrixIntegrator()
            await integrator.start()
            
            # Test entropy analysis
            entropy_results = await integrator.analyze_hrm_entropy(hrm_model)
            print(f"✅ Entropy analysis completed: {entropy_results}")
            
            await integrator.stop()
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    asyncio.run(test())
