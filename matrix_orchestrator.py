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

# ========= Contracts =========
class MatrixChunk(BaseModel):
    id: str
    data: List[List[float]]  # row-major
    meta: Dict[str, Any] = Field(default_factory=dict)

class PolySpec(BaseModel):
    degree: int = 3
    basis: str = "chebyshev"  # "chebyshev" | "legendre" | "monomial"

class EntropyReport(BaseModel):
    chunk_id: str
    shannon: float
    spectral: float
    compression_ratio: float

class AdjacencyPayload(BaseModel):
    labels: List[str]
    adjacency: List[List[float]]
    mode: str = "kfp"
    beta: float = 0.8

class OptimizeRequest(BaseModel):
    matrix: List[List[float]]
    method: str = "sparsity"  # "sparsity" | "rank" | "structure" | "poly"
    params: Dict[str, Any] = Field(default_factory=dict)
    adjacency: Optional[AdjacencyPayload] = None

class OptimizeResponse(BaseModel):
    objective: float
    matrix_opt: List[List[float]]
    iterations: int
    meta: Dict[str, Any] = Field(default_factory=dict)

class RunPlan(BaseModel):
    run_id: str
    chunks: List[MatrixChunk]
    poly: PolySpec
    optimize: OptimizeRequest
    export_path: str = "results.json"
    # optional feature toggles / inputs
    ccl_report_path: Optional[str] = None
    stability_metrics_path: Optional[str] = None
    sim_trials: int = 8
    sim_degrees: List[int] = Field(default_factory=lambda: [2, 3, 4, 5])

# ========= Utils =========
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj

def sha1(obj: Any) -> str:
    return hashlib.sha1(json.dumps(_jsonable(obj), sort_keys=True, default=_jsonable).encode()).hexdigest()

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))

def load_json(path: Path) -> Optional[Any]:
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
            return {"ok": bool(data.get("ok", True)), "data": data, "fallback": True}
    
    def is_degraded(self) -> bool:
        """Check if backend is in degraded mode"""
        return self._circuit_open

    async def analyze_entropy(self, chunk: MatrixChunk) -> EntropyReport:
        try:
            data = await self._post("/entropy", {"data": chunk.data})
            return EntropyReport(chunk_id=chunk.id, **data)
        except Exception as e:
            log.warning("Primary entropy endpoint failed, trying dispatcher fallback: %s", e)
            try:
                data = await self._dispatch("entropy", {"data": chunk.data})
                if "error" not in data:
                    return EntropyReport(chunk_id=chunk.id, **data)
            except Exception as e2:
                log.warning("Dispatcher fallback also failed: %s", e2)
            
            log.warning("Falling back to local entropy for chunk %s", chunk.id)
            return entropy_local(chunk)

    async def project_chebyshev(self, matrix, spec: PolySpec) -> List[List[float]]:
        payload = {"matrix": matrix, "degree": spec.degree, "basis": spec.basis}
        try:
            data = await self._post("/chebyshev_project", payload)
            return data["matrix"]
        except Exception as e:
            log.warning("Primary chebyshev endpoint failed, trying dispatcher fallback: %s", e)
            try:
                data = await self._dispatch("chebyshev_project", matrix, spec.degree, spec.basis)
                if "error" not in data and "matrix" in data:
                    return data["matrix"]
            except Exception as e2:
                log.warning("Dispatcher fallback also failed: %s", e2)
            
            log.warning("Falling back to local chebyshev projection")
            return chebyshev_local(matrix, spec)

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
            data = await self._post("/coherence", code_signature)
        except Exception as e:
            log.warning("Primary coherence endpoint failed, trying dispatcher fallback: %s", e)
            data = await self._dispatch("coherence", code_signature)
            if "error" in data:
                log.warning("Ghost score fallback (default 0.1): %s", e)
                return 0.1
        return float(data.get("ghost_score", 0.1))

class MockBackend:
    async def aclose(self) -> None:
        return None

    async def analyze_entropy(self, chunk: MatrixChunk) -> EntropyReport:
        return entropy_local(chunk)

    async def project_chebyshev(self, matrix, spec: PolySpec) -> List[List[float]]:
        return chebyshev_local(matrix, spec)

    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        mat = np.array(req.matrix, dtype=float)
        return OptimizeResponse(
            objective=float(np.linalg.norm(mat, ord=1)),
            matrix_opt=mat.tolist(),
            iterations=3,
            meta={"mock": True}
        )

    async def ghost_score(self, code_signature: Dict[str, Any]) -> float:
        return 0.1

def get_backend() -> Backend:
    if SET.backend.lower() == "julia":
        log.info("Using Julia backend at %s", SET.julia_url)
        return JuliaBackend(SET.julia_url)
    log.info("Using Mock backend")
    return MockBackend()

# ========= Local fallback math =========
def entropy_local(chunk: MatrixChunk) -> EntropyReport:
    arr = np.nan_to_num(np.asarray(chunk.data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return EntropyReport(chunk_id=chunk.id, shannon=0.0, spectral=0.0, compression_ratio=1.0)

    # Shannon entropy on normalized histogram of values
    hist, _ = np.histogram(arr, bins=64, density=True)
    p = hist[hist > 0]
    shannon = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    # spectral entropy using normalized singular values
    s = np.linalg.svd(arr, compute_uv=False)
    s_sum = s.sum()
    p_s = s / s_sum if s_sum > 0 else s
    spectral = float(-(p_s * np.log2(p_s + 1e-12)).sum()) if p_s.size else 0.0

    # compression proxy: low-rankness indicator
    rank = np.linalg.matrix_rank(arr) if arr.size else 1
    compression_ratio = float(arr.size / max(rank, 1))
    return EntropyReport(chunk_id=chunk.id, shannon=shannon, spectral=spectral, compression_ratio=compression_ratio)

def chebyshev_local(matrix: List[List[float]], spec: PolySpec) -> List[List[float]]:
    x = np.asarray(matrix, dtype=float)
    if x.size == 0:
        return x.tolist()

    denom = np.abs(x).max()
    scale = 1.0 / (denom + 1e-9)
    z = np.clip(x * scale, -1.0, 1.0)

    acc = np.zeros_like(z)
    for k in range(spec.degree + 1):
        # T_k(z) = cos(k arccos z)
        acc += np.cos(k * np.arccos(z))
    acc /= (spec.degree + 1)
    # rescale back roughly; keep bounded for stability
    return np.clip(acc / max(scale, 1e-9), -1e6, 1e6).tolist()

# ========= DAG Orchestrator =========
@dataclass
class Node:
    name: str
    fn: Callable[..., Any]
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

        async def run_with_deps(name: str):
            for d in self.nodes[name].deps:
                if d not in ctx:
                    ctx[d] = await run_with_deps(d)
            ctx[name] = await self._run_node(run_id, name, ctx)
            return ctx[name]

        for n in order:
            await run_with_deps(n)
        return ctx

# ========= Wire the pipeline =========
backend = get_backend()
orch = Orchestrator(backend)

@orch.node("entropy_map", deps=["chunks"])
async def entropy_map(ctx: Dict[str, Any]):
    chunks: List[MatrixChunk] = ctx["chunks"]
    reports = await asyncio.gather(*[backend.analyze_entropy(c) for c in chunks])
    # warn if any chunk is too "hot"
    hot = [r for r in reports if r.shannon > SET.entropy_warn_threshold]
    if hot:
        log.warning("⚠️  %d chunks show high entropy (>%.2f)", len(hot), SET.entropy_warn_threshold)
    return [r.model_dump() for r in reports]

@orch.node("build_adjacency", deps=["entropy_map"])
async def build_adjacency(ctx: Dict[str, Any]):
    # Expect ctx["ccl_report_path"] -> str (optional)
    rpt_path = ctx.get("ccl_report_path")
    if not rpt_path:
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
        
        # Simple benchmark with random matrices
        import time
        payloads = [
            {"function": "optimize_matrix", "args": [np.random.rand(32, 32).tolist(), "sparsity"]} 
            for _ in range(8)
        ]
        
        times = []
        for p in payloads:
            t0 = time.perf_counter()
            try:
                await backend._dispatch("optimize_matrix", p["args"][0], p["args"][1])
                dt = time.perf_counter() - t0
                times.append(dt)
            except Exception as e:
                log.warning("Benchmark request failed: %s", e)
                continue
        
        if not times:
            return {"n": 0, "p50": 0.0, "p95": 0.0, "error": "all requests failed"}
        
        return {
            "n": len(times),
            "p50": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
            "mean": float(np.mean(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times))
        }
        
    except Exception as e:
        log.warning("Benchmark failed: %s", e)
        return {"n": 0, "p50": 0.0, "p95": 0.0, "error": str(e)}

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
    optimize_req: OptimizeRequest = ctx["optimize"]
    stability = ctx.get("stability_probe", {})
    
    signature = {
        "modules": ["matrix_orchestrator"],
        "nodes": list(orch.nodes.keys()),
        "policy": optimize_req.method,
        "stability_prior": stability.get("mean_stability", 0.0) if stability else 0.0,
    }
    
    score = await backend.ghost_score(signature)
    
    # Adjust threshold based on stability metrics
    threshold = SET.ghost_score_max
    if stability and stability.get("mean_stability", 0.0) > 0.8:
        # High stability allows slightly higher ghost scores
        threshold *= 1.2
        log.info("Stability-adjusted threshold: %.3f (stability: %.3f)", threshold, stability["mean_stability"])
    
    if score > threshold:
        raise RuntimeError(f"Ghost score {score:.2f} exceeds threshold {threshold:.3f}. Aborting.")
    
    return {"ghost_score": score, "threshold": threshold, "stability_metrics": stability}

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
    out_path = Path(SET.run_root) / run_id / ctx.get("export_path", "results.json")
    
    # Include all available data
    payload = {
        "run_id": run_id,
        "entropy": ctx["entropy_map"],
        "optimize_result": ctx["optimize"],   # optimized result, not request
        "poly_spec": ctx["poly"].model_dump(),
        "coherence_gate": ctx.get("coherence_gate", {}),
        "build_adjacency": ctx.get("build_adjacency", {}),
        "stability_probe": ctx.get("stability_probe", {}),
        "simulate_plan": ctx.get("simulate_plan", {}),
        "backend_health": ctx.get("backend_health", {}),
        "bench_backend": ctx.get("bench_backend", {}),
    }
    
    # Check if backend is degraded
    if isinstance(backend, JuliaBackend) and backend.is_degraded():
        payload["degraded"] = True
        log.warning("Exporting degraded results due to backend failures")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    
    # Also write a UI-friendly version
    ui_path = out_path.parent / "ui.json"
    stability_probe = ctx.get("stability_probe")
    coherence_gate = ctx.get("coherence_gate", {})
    
    ui_payload = {
        "run_id": run_id,
        "objective": ctx["optimize"].get("objective", 0.0),
        "iterations": ctx["optimize"].get("iterations", 0),
        "ghost_score": coherence_gate.get("ghost_score", 0.0) if coherence_gate else 0.0,
        "stability": stability_probe.get("mean_stability", 0.0) if stability_probe else 0.0,
        "hot_chunks": len([r for r in ctx["entropy_map"] if r.get("shannon", 0) > SET.entropy_warn_threshold])
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
        "export_path": plan.export_path,
        "ccl_report_path": plan.ccl_report_path,
        "stability_metrics_path": plan.stability_metrics_path,
        "sim_trials": plan.sim_trials,
        "sim_degrees": plan.sim_degrees,
    }
    
    # Dynamic order based on available features
    order = ["backend_health", "entropy_map"]
    
    # Optional nodes
    if ctx.get("ccl_report_path"):
        order.append("build_adjacency")
    if ctx.get("stability_metrics_path"):
        order.append("stability_probe")
    
    # Core pipeline
    order.extend(["poly_project", "coherence_gate", "optimize"])
    
    # Optional simulation
    if ctx.get("sim_trials") and ctx.get("sim_degrees"):
        order.append("simulate_plan")
    
    # Always include benchmark for performance tracking
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
