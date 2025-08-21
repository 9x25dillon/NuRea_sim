#!/usr/bin/env python3
"""
NuRea Orchestrator - CLI for Matrix Optimization Pipeline
Integrates Python orchestration with Julia backend and ChaosRAGJulia
"""

import os
import json
import argparse
import csv
import random
import logging
from typing import List, Dict, Any
import requests
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("ORCH_LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("orchestrator")

def load_matrix(step: Dict[str, Any]) -> List[List[float]]:
    """Load matrix from file or generate synthetic data"""
    src = step.get("source")
    shape = step.get("shape", [10, 10])
    seed = step.get("seed")
    
    if src and os.path.exists(src):
        log.info(f"Loading matrix from {src}")
        df = pd.read_csv(src, header=None)
        return df.values.tolist()
    
    # Generate synthetic matrix if file missing
    rows, cols = shape
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        log.info(f"Generating {rows}x{cols} synthetic matrix with seed {seed}")
    
    return np.random.normal(0.0, 1.0, (rows, cols)).tolist()

def save_matrix(M: List[List[float]], target: str):
    """Save matrix to CSV file"""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    df = pd.DataFrame(M)
    df.to_csv(target, index=False, header=False)
    log.info(f"Saved matrix to {target}")

def optimize_matrix_julia(M: List[List[float]], backend_url: str, method: str, **params) -> List[List[float]]:
    """Send matrix to Julia backend for optimization"""
    payload = {
        "matrix": M,
        "method": method,
        "params": params
    }
    
    log.info(f"Optimizing matrix via Julia backend: {method}")
    r = requests.post(f"{backend_url}/optimize", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    log.info(f"Julia optimization complete: {data.get('rows', '?')}x{data.get('cols', '?')}")
    
    return data["optimized_matrix"]

def vectorize_matrix_chaosrag(M: List[List[float]], backend_url: str, method: str, dimensions: int) -> List[List[float]]:
    """Send matrix to ChaosRAGJulia for vectorization"""
    # Flatten matrix for vectorization
    flat_matrix = [item for sublist in M for item in sublist]
    
    payload = {
        "vectors": [flat_matrix],
        "method": method,
        "target_dimensions": dimensions
    }
    
    log.info(f"Vectorizing matrix via ChaosRAGJulia: {method} -> {dimensions}D")
    r = requests.post(f"{backend_url}/vectorize", json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    log.info(f"Vectorization complete: {len(data.get('vectors', [[]])[0])} dimensions")
    
    # Return the vectorized matrix (reshape if needed)
    return data["vectors"]

def execute_pipeline(plan: Dict[str, Any]) -> None:
    """Execute the optimization pipeline"""
    backend = plan["backend"]
    pipeline = plan["pipeline"]
    
    M = None
    for i, step in enumerate(pipeline):
        op = step["op"]
        log.info(f"Pipeline step {i+1}/{len(pipeline)}: {op}")
        
        if op == "load_matrix":
            M = load_matrix(step)
            log.info(f"Loaded matrix: {len(M)}x{len(M[0])}")
            
        elif op == "optimize":
            if step.get("backend") == "julia":
                M = optimize_matrix_julia(
                    M, 
                    backend["julia_url"], 
                    step.get("method", "chebyshev_projection"),
                    **{k: v for k, v in step.items() if k not in ["op", "method", "backend"]}
                )
            else:
                log.warning(f"Unknown optimization backend: {step.get('backend')}")
                
        elif op == "vectorize":
            if step.get("backend") == "chaos_rag":
                M = vectorize_matrix_chaosrag(
                    M,
                    backend["chaos_rag_url"],
                    step.get("method", "l2_normalize"),
                    step.get("dimensions", 1536)
                )
            else:
                log.warning(f"Unknown vectorization backend: {step.get('backend')}")
                
        elif op == "save":
            save_matrix(M, step["target"])
            
        else:
            raise SystemExit(f"Unknown operation: {op}")
    
    log.info("Pipeline execution complete!")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="NuRea Matrix Optimization Orchestrator")
    parser.add_argument("--plan", default=os.getenv("PLAN_PATH", "/app/plan.json"))
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    log.info(f"NuRea Orchestrator starting with plan: {args.plan}")
    
    try:
        with open(args.plan, "r") as f:
            plan = json.load(f)
        
        log.info(f"Loaded plan: {plan.get('run_id', 'unknown')}")
        execute_pipeline(plan)
        
    except FileNotFoundError:
        log.error(f"Plan file not found: {args.plan}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in plan file: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
