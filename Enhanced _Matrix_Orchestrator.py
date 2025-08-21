#!/usr/bin/env python3
"""
Enhanced Matrix Orchestrator - Declarative DAG with Typed Contracts
Integrates with au'La/LA Julia server for mathematical optimization
"""

from __future__ import annotations
import asyncio
import json

import os
import time
import random
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Callable

import httpx
import numpy as np
from pydantic import BaseModel, Field

# Fallback for pydantic-settings if not installed
# NOTE: For full env var support (MATRIX_BACKEND, MATRIX_JULIA_URL, etc.), install:
#   pip install pydantic-settings>=2
# Without it, Settings will use defaults and you can override via CLI/JSON plan
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # fallback if the package isn't installed
    from pydantic import BaseModel as BaseSettings  # type: ignore
    class SettingsConfigDict(dict):  # shim so your Settings.model_config line still works
        pass


# ========= Logging =========
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("matrix_orchestrator")

# ========= Config =========
class Settings(BaseSettings):
    run_root: str = "./runs"
    backend: str = "mock"            # "julia" | "mock"
    julia_url: str = "http://localhost:9000"
    max_concurrency: int = 4
    request_timeout_s: float = 60.0
    retry_attempts: int = 3
    retry_backoff_s: float = 0.5
    entropy_warn_threshold: float = 0.85
    ghost_score_max: float = 0.35     # CCL coherence gate

    model_config = SettingsConfigDict(env_prefix="MATRIX_")

SET = Settings()

# ========= Models =========
class MatrixChunk(BaseModel):
    id: str
    data: List[List[float]]  # row-major
    meta: Dict[str, Any] = Field(default_factory=dict)

class PolySpec(BaseModel):
    degree: int
    basis: str = "chebyshev"

class EntropyReport(BaseModel):
    shannon: float
    surprise: float = 0.0
    complexity: float = 0.0

class OptimizeRequest(BaseModel):
    matrix: List[List[float]]
    method: str = "gradient"
    params: Dict[str, Any] = Field(default_factory=dict)
    adjacency: Optional['AdjacencyPayload'] = None

class OptimizeResponse(BaseModel):
    optimized: List[List[float]]
    loss: float = 0.0
    iterations: int = 0
    converged: bool = False

class AdjacencyPayload(BaseModel):
    labels: List[str]
    adjacency: List[List[float]]
    mode: str = "kfp"
    beta: float = 0.8

class RunPlan(BaseModel):
    run_id: str
    chunks: List[MatrixChunk]
    poly: PolySpec
    optimize: OptimizeRequest

# ========= Utility functions =========
def sha1(data: Any) -> str:
    content = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(content).hexdigest()[:8]

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return None

def cache_path(run_id: str, node: str, key: str) -> Path:
    return Path(SET.run_root) / run_id / node / f"{key}.json"

def stable_cache_key_from_deps(ctx: Dict[str, Any], deps: List[str], extra: Optional[List[str]] = None) -> str:
    keys = list(deps) + (extra or [])
    material = {k: ctx.get(k) for k in keys}
    material["_run_id"] = ctx.get("run_id")
    return sha1(material)

def backoff_sleep(attempt: int, base: float) -> float:
    # Exponential backoff with jitter
    return base * (2 ** attempt) * (0.5 + random.random())

# ========= Backend Protocols =========
class Backend(Protocol):
    async def analyze_entropy(self, chunk: MatrixChunk) -> EntropyReport: ...
    async def project_chebyshev(self, matrix: List[List[float]], spec: PolySpec) -> List[List[float]]: ...
    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse: ...
    async def ghost_score(self, code_signature: Dict[str, Any]) -> float: ...
    async def aclose(self) -> None: ...

# ========= Backend Implementations =========
class JuliaBackend:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=SET.request_timeout_s)
        self._failure_count = 0
        self._max_failures = 3  # Circuit breaker threshold
        self._circuit_open = False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._circuit_open:
            log.warning("Circuit breaker open, skipping request to %s", path)
            return {"error": "circuit_breaker_open", "degraded": True}
        
        last_exc = None
        for attempt in range(SET.retry_attempts):
            try:
                r = await self._client.post(f"{self.base_url}{path}", json=payload)
                r.raise_for_status()
                # Reset failure count on success
                self._failure_count = 0
                return r.json()
            except Exception as e:
                last_exc = e
                sleep_s = backoff_sleep(attempt, SET.retry_backoff_s)
                log.warning("POST %s failed (attempt %d/%d): %s; retrying in %.2fs",
                            path, attempt + 1, SET.retry_attempts, e, sleep_s)
                await asyncio.sleep(sleep_s)
        
        # Increment failure count
        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._circuit_open = True
            log.error("Circuit breaker opened after %d consecutive failures", self._failure_count)
        
        assert last_exc is not None
        raise last_exc

    async def _dispatch(self, function: str, *args):
        """Fallback dispatcher for FastAPI mocks that expose single endpoint"""
        try:
            return await self._post("/", {"function": function, "args": list(args)})
        except Exception as e:
            log.warning("Dispatch fallback failed: %s", e)
            return {"error": "dispatch failed"}

    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        try:
            data = await self._post("/health", {})
            return {"ok": bool(data.get("ok", True)), "data": data}
        except Exception:
            # Try dispatcher fallback
            data = await self._dispatch("ping")
            return {"ok": bool(data.get("ok", True)), "data": data}

    async def analyze_entropy(self, chunk: MatrixChunk) -> EntropyReport:
        try:
            data = await self._post("/entropy", chunk.model_dump())
        except Exception as e:
            log.warning("Primary entropy endpoint failed, trying dispatcher fallback: %s", e)
            data = await self._dispatch("analyze_entropy", chunk.model_dump())
            if "error" in data:
                # Fallback to local calculation
                return entropy_local(chunk)
        return EntropyReport(**data)

    async def project_chebyshev(self, matrix: List[List[float]], spec: PolySpec) -> List[List[float]]:
        try:
            data = await self._post("/chebyshev", {"matrix": matrix, "spec": spec.model_dump()})
        except Exception as e:
            log.warning("Primary chebyshev endpoint failed, trying dispatcher fallback: %s", e)
            data = await self._dispatch("project_chebyshev", matrix, spec.model_dump())
            if "error" in data:
                # Fallback to local calculation
                return chebyshev_local(matrix, spec)
        return data.get("projected", matrix)

    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        try:
            data = await self._post("/optimize", req.model_dump())
        except Exception as e:
            log.warning("Primary optimize endpoint failed, trying dispatcher fallback: %s", e)
            data = await self._dispatch("optimize", req.model_dump())
            if "error" in data:
                raise RuntimeError(f"Optimize failed: {data['error']}")
        return OptimizeResponse(**data)

    async def ghost_score(self, code_signature: Dict[str, Any]) -> float:
        try:
            data = await self._post("/ghost", {"signature": code_signature})
        except Exception as e:
            log.warning("Primary ghost endpoint failed, trying dispatcher fallback: %s", e)
            data = await self._dispatch("ghost_score", code_signature)
            if "error" in data:
                log.warning("Ghost score failed, returning safe value: %s", data["error"])
                return 0.0
        return float(data.get("score", 0.0))

class MockBackend:
    async def aclose(self) -> None:
        pass

    async def analyze_entropy(self, chunk: MatrixChunk) -> EntropyReport:
        # Simple mock entropy calculation
        data = np.array(chunk.data)
        if data.size == 0:
            return EntropyReport(shannon=0.0)
        
        # Normalize and calculate shannon entropy
        data_flat = data.flatten()
        data_norm = np.abs(data_flat) / (np.sum(np.abs(data_flat)) + 1e-8)
        entropy = -np.sum(data_norm * np.log2(data_norm + 1e-8))
        return EntropyReport(shannon=float(entropy))

    async def project_chebyshev(self, matrix: List[List[float]], spec: PolySpec) -> List[List[float]]:
        # Mock projection - just return the matrix with slight modification
        np_matrix = np.array(matrix)
        if np_matrix.size == 0:
            return matrix
        # Apply a simple polynomial-like transformation
        degree = spec.degree
        projected = np_matrix * (degree / (degree + 1))
        return projected.tolist()

    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        # Mock optimization - return the matrix with small random perturbations
        np_matrix = np.array(req.matrix)
        if np_matrix.size == 0:
            return OptimizeResponse(optimized=req.matrix)
        
        # Add small optimization-like changes
        optimized = np_matrix + np.random.normal(0, 0.01, np_matrix.shape)
        return OptimizeResponse(
            optimized=optimized.tolist(),
            loss=0.1,
            iterations=10,
            converged=True
        )

    async def ghost_score(self, code_signature: Dict[str, Any]) -> float:
        # Mock ghost score - return a safe value
        return 0.1

def get_backend() -> Backend:
    if SET.backend == "julia":
        return JuliaBackend(SET.julia_url)
    else:
        return MockBackend()

# ========= Local fallback math =========
def entropy_local(chunk: MatrixChunk) -> EntropyReport:
    """Local fallback for entropy calculation when backend fails"""
    data = np.array(chunk.data)
    if data.size == 0:
        return EntropyReport(shannon=0.0)
    
    data_flat = data.flatten()
    data_norm = np.abs(data_flat) / (np.sum(np.abs(data_flat)) + 1e-8)
    entropy = -np.sum(data_norm * np.log2(data_norm + 1e-8))
    return EntropyReport(shannon=float(entropy))

def chebyshev_local(matrix: List[List[float]], spec: PolySpec) -> List[List[float]]:
    """Local fallback for Chebyshev projection when backend fails"""
    np_matrix = np.array(matrix)
    if np_matrix.size == 0:
        return matrix
    
    # Simple polynomial approximation
    degree = spec.degree
    projected = np_matrix * (degree / (degree + 1))
    return projected.tolist()

# ========= DAG Orchestrator =========
@dataclass
class Node:
    name: str
    fn: Callable
    deps: List[str]

class Orchestrator:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.nodes: Dict[str, Node] = {}
        self.sem = asyncio.Semaphore(SET.max_concurrency)

    def node(self, name: str, deps: List[str]):
        def deco(fn):
            self.nodes[name] = Node(name=name, fn=fn, deps=deps)
            return fn
        return deco

    async def _run_node(self, run_id: str, name: str, ctx: Dict[str, Any]) -> Any:
        node = self.nodes[name]
        extra = getattr(node.fn, "_extra_cache_keys", [])
        key = stable_cache_key_from_deps(ctx, node.deps, extra=extra)
        path = cache_path(run_id, name, key)

        if path.exists():
            log.info("[%s] cache hit", name)
            return load_json(path)

        async with self.sem:
            t0 = time.perf_counter()
            out = await node.fn(ctx)
            save_json(path, out)
            dt = time.perf_counter() - t0
            log.info("[%s] %.2fs", name, dt)
            return out

    async def run(self, run_id: str, order: List[str], seed_ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(seed_ctx)
        ctx["run_id"] = run_id
        
        for name in order:
            if name not in self.nodes:
                log.warning("Node %s not found, skipping", name)
                continue
            
            ctx[name] = await self._run_node(run_id, name, ctx)
        
        return ctx

# ========= Wire the pipeline =========
backend = get_backend()
orch = Orchestrator(backend)

@orch.node("entropy_map", deps=["chunks"])
async def entropy_map(ctx: Dict[str, Any]):
    chunks = ctx["chunks"]
    reports = []
    for chunk in chunks:
        report = await backend.analyze_entropy(chunk)
        reports.append({
            "chunk_id": chunk.id,
            "shannon": report.shannon,
            "surprise": report.surprise,
            "complexity": report.complexity
        })
    return reports

@orch.node("poly_project", deps=["chunks", "poly"])
async def poly_project(ctx: Dict[str, Any]):
    spec: PolySpec = ctx["poly"]
    # Stack chunks vertically into one matrix
    mats = [np.asarray(c.data, dtype=float) for c in ctx["chunks"]]
    mat = np.vstack(mats) if mats else np.zeros((0, 0), dtype=float)
    out = await backend.project_chebyshev(mat.tolist(), spec)
    return {"projected": out}

@orch.node("coherence_gate", deps=["poly_project", "stability_probe"])
async def coherence_gate(ctx: Dict[str, Any]):
    proj = ctx["poly_project"]["projected"]
    stability = ctx.get("stability_probe")
    
    # Calculate ghost score from projection signature
    signature = {
        "shape": [len(proj), len(proj[0]) if proj else 0],
        "norm": float(np.linalg.norm(proj)) if proj else 0.0,
        "mean": float(np.mean(proj)) if proj else 0.0
    }
    score = await backend.ghost_score(signature)
    
    # Bias threshold based on stability metrics
    threshold = SET.ghost_score_max
    if stability and stability.get("mean_stability"):
        # Lower threshold if trainer is unstable (allow more ghost)
        instability = 1.0 - stability["mean_stability"]
        threshold += instability * 0.1
    
    if score > threshold:
        raise RuntimeError(f"Ghost score {score:.2f} exceeds threshold {threshold:.3f}. Aborting.")
    
    return {"ghost_score": score, "threshold": threshold, "stability_metrics": stability}

@orch.node("build_adjacency", deps=["entropy_map"])
async def build_adjacency(ctx: Dict[str, Any]):
    # Build adjacency matrix from CCL report if available
    rpt_path = ctx.get("ccl_report_path")
    if not rpt_path or not Path(rpt_path).exists():
        return None
    
    try:
        rep = json.loads(Path(rpt_path).read_text())
        fqs = [f["qualname"] for f in rep.get("functions", []) if f.get("arity", 0) == 1]
        idx = {q: i for i, q in enumerate(fqs)}
        n = len(fqs)
        adj = [[0.0] * n for _ in range(n)]
        
        for p in rep.get("pipelines", []):
            if not isinstance(p.get("pair", []), list) or len(p["pair"]) != 2:
                continue
            qf, qg = p["pair"]
            if qf in idx and qg in idx:
                i, j = idx[qf], idx[qg]
                try: 
                    agr = float(p.get("agreement_rate", 0.0))
                except: 
                    agr = 0.0
                w = max(0.0, 1.0 - agr)
                adj[i][j] = max(adj[i][j], w)
                adj[j][i] = max(adj[j][i], w)
        
        return {"adjacency": {"labels": fqs, "adjacency": adj, "mode": "kfp", "beta": 0.8}}
    except Exception as e:
        log.warning("Failed to build adjacency from CCL report: %s", e)
        return None

# Add extra cache keys for build_adjacency
build_adjacency._extra_cache_keys = ["ccl_report_path"]

@orch.node("simulate_plan", deps=["chunks", "poly"])
async def simulate_plan(ctx: Dict[str, Any]):
    rng = np.random.default_rng(0)  # Deterministic for cache validation
    trials = ctx.get("sim_trials", 8)
    degrees = ctx.get("sim_degrees", [2, 3, 4, 5])
    results = []
    
    for d in degrees:
        mat = np.vstack([np.asarray(c.data) for c in ctx["chunks"]])
        if mat.size == 0:
            continue
            
        noise = rng.normal(scale=0.01, size=mat.shape)
        spec = PolySpec(degree=d, basis=ctx["poly"].basis)
        proj = await backend.project_chebyshev((mat + noise).tolist(), spec)
        results.append({
            "degree": d, 
            "proj_norm1": float(np.abs(proj).sum()),
            "noise_scale": 0.01
        })
    
    return {"sims": results, "trials": trials, "degrees": degrees}

# Add extra cache keys for simulate_plan
simulate_plan._extra_cache_keys = ["sim_trials", "sim_degrees"]

@orch.node("backend_health", deps=[])
async def backend_health(ctx: Dict[str, Any]):
    """Check backend health for readiness gates"""
    try:
        if isinstance(backend, JuliaBackend):
            data = await backend.health_check()
            ok = bool(data.get("ok", True))
            if not ok:
                raise RuntimeError("Backend health failed")
            return {"ok": ok, "data": data}
        else:
            # Mock backend is always healthy
            return {"ok": True, "data": {"backend": "mock"}}
    except Exception as e:
        log.warning("Health check failed: %s", e)
        return {"ok": False, "error": str(e)}

@orch.node("stability_probe", deps=[])
async def stability_probe(ctx: Dict[str, Any]):
    """Consume trainer stability metrics and bias the coherence gate"""
    # Expect an optional path that the trainer writes
    path = ctx.get("stability_metrics_path")
    if not path or not Path(path).exists():
        return None
    
    try:
        m = json.loads(Path(path).read_text())
        return {
            "mean_stability": m.get("mean_stability_score", 0.0),
            "logit_stability": m.get("logit_stability", 0.0),
            "stability_variance": m.get("stability_variance", 0.0)
        }
    except Exception as e:
        log.warning("Failed to read stability metrics: %s", e)
        return None

# Add extra cache keys for stability_probe
stability_probe._extra_cache_keys = ["stability_metrics_path"]

@orch.node("bench_backend", deps=[])
async def bench_backend(ctx: Dict[str, Any]):
    """Benchmark backend API performance"""
    try:
        if not isinstance(backend, JuliaBackend):
            return {"n": 0, "p50": 0.0, "p95": 0.0, "backend": "mock"}
        
        # Run a few benchmark requests
        times = []
        test_chunk = MatrixChunk(id="bench", data=[[1.0, 2.0], [3.0, 4.0]])
        
        for _ in range(5):
            t0 = time.perf_counter()
            await backend.analyze_entropy(test_chunk)
            dt = time.perf_counter() - t0
            times.append(dt)
        
        times.sort()
        return {
            "n": len(times),
            "p50": times[len(times)//2],
            "p95": times[int(len(times)*0.95)],
            "backend": "julia"
        }
    except Exception as e:
        log.warning("Backend benchmark failed: %s", e)
        return {"n": 0, "p50": 0.0, "p95": 0.0, "error": str(e)}

@orch.node("optimize", deps=["coherence_gate", "build_adjacency"])
async def optimize(ctx: Dict[str, Any]):
    req: OptimizeRequest = ctx["optimize"]
    proj = ctx["poly_project"]["projected"]
    adj = ctx.get("build_adjacency", None)
    
    # set projected matrix without mutating incoming ctx object structure
    exec_req = OptimizeRequest(
        matrix=proj, 
        method=req.method, 
        params=req.params,
        adjacency=(
            AdjacencyPayload(**adj["adjacency"])
            if (req.method == "structure" and adj and "adjacency" in adj) else None
        ),
    )
    resp = await backend.optimize(exec_req)
    return resp.model_dump()

@orch.node("export", deps=["optimize", "poly", "entropy_map"])
async def export(ctx: Dict[str, Any]):
    run_id = ctx["run_id"]
    out_path = Path(SET.run_root) / run_id / "export.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export results
    export_data = {
        "run_id": run_id,
        "optimized_matrix": ctx["optimize"]["optimized"],
        "loss": ctx["optimize"]["loss"],
        "poly_spec": ctx["poly"].model_dump(),
        "entropy_reports": ctx["entropy_map"],
        "timestamp": time.time()
    }
    save_json(out_path, export_data)
    
    # Generate UI payload
    ui_path = Path(SET.run_root) / run_id / "ui_payload.json"
    ui_payload = {
        "run_id": run_id,
        "status": "completed",
        "loss": ctx["optimize"]["loss"],
        "converged": ctx["optimize"]["converged"],
        "iterations": ctx["optimize"]["iterations"],
        "entropy_stats": {
            "mean": np.mean([r["shannon"] for r in ctx["entropy_map"]]),
            "max": max([r["shannon"] for r in ctx["entropy_map"]]),
            "hot_chunks": len([r for r in ctx["entropy_map"] if r.get("shannon", 0) > SET.entropy_warn_threshold])
        }
    }
    ui_path.write_text(json.dumps(ui_payload, indent=2))
    
    # Write manifest for debugging
    manifest_path = out_path.parent / "manifest.json"
    manifest = {
        "run_id": run_id,
        "timestamp": time.time(),
        "nodes": {}
    }
    
    # Collect cache keys and paths for each node
    for node_name in orch.nodes:
        if node_name in ctx:
            node = orch.nodes[node_name]
            extra = getattr(node.fn, "_extra_cache_keys", [])
            key = stable_cache_key_from_deps(ctx, node.deps, extra=extra)
            cache_file = cache_path(run_id, node_name, key)
            manifest["nodes"][node_name] = {
                "deps": node.deps,
                "extra_cache_keys": extra,
                "cache_key": key,
                "cache_file": str(cache_file),
                "exists": cache_file.exists()
            }
    
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    return {"path": str(out_path), "ui_path": str(ui_path), "manifest_path": str(manifest_path)}

# ========= Entry points =========
async def orchestrate(plan: RunPlan) -> Dict[str, Any]:
    ctx = {
        "run_id": plan.run_id,
        "chunks": plan.chunks,
        "poly": plan.poly,
        "optimize": plan.optimize,
    }
    
    # Build execution order
    order = ["backend_health"]
    
    # Add stability probe if metrics are available
    if os.environ.get("STABILITY_METRICS_PATH"):
        ctx["stability_metrics_path"] = os.environ["STABILITY_METRICS_PATH"]
        order.append("stability_probe")
    
    order.extend([
        "entropy_map",
        "poly_project", 
        "coherence_gate",
        "build_adjacency",
        "optimize"
    ])
    
    # Optional simulation and benchmarking
    if plan.optimize.params.get("run_simulations", False):
        ctx.update(plan.optimize.params)  # sim_trials, sim_degrees
        order.append("simulate_plan")
    
    order.append("bench_backend")
    
    order.append("export")
    
    try:
        result = await orch.run(plan.run_id, order, ctx)
        return result
    finally:
        # ensure backend cleans up sockets
        try:
            await backend.aclose()
        except Exception:
            pass

def run_cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to plan.json")
    args = ap.parse_args()
    plan = RunPlan(**json.loads(Path(args.plan).read_text()))
    out = asyncio.run(orchestrate(plan))
    
    # Convert Pydantic models to dicts for JSON serialization
    def make_serializable(obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif isinstance(obj, (MatrixChunk, PolySpec, OptimizeRequest)):
            return obj.model_dump() if hasattr(obj, 'model_dump') else str(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    serializable_out = make_serializable(out)
    print(json.dumps(serializable_out, indent=2))

# ========= Legacy compatibility =========
class MatrixOrchestrator:
    """Legacy compatibility wrapper for existing code"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.backend = get_backend()
        self.orch = Orchestrator(self.backend)

    async def start(self):
        log.info("🚀 Enhanced Matrix Orchestrator started")
        log.info("Backend: %s", SET.backend)
        log.info("Julia URL: %s", SET.julia_url)
        return True

    async def stop(self):
        log.info("🛑 Enhanced Matrix Orchestrator stopped")
        await self.backend.aclose()
        return True

    async def health_check(self):
        try:
            test_chunk = MatrixChunk(id="test", data=[[1.0, 2.0], [3.0, 4.0]])
            await self.backend.analyze_entropy(test_chunk)
            return {"status": "healthy", "backend": SET.backend}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def run_pipeline(
        self,
        run_id: str,
        chunks: List[MatrixChunk],
        poly_spec: Dict[str, Any],
        optimize_req: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan = RunPlan(
            run_id=run_id,
            chunks=chunks,
            poly=PolySpec(**poly_spec),
            optimize=OptimizeRequest(**optimize_req),
        )
        return await orchestrate(plan)

if __name__ == "__main__":
    run_cli()