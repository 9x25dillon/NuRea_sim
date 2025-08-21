#!/usr/bin/env python3
"""
Categorical Coherence Linter (CCL)
Entropy‑driven "ghost in the code" detector

What it does
------------
• Parses a Python codebase (file or directory) and builds a light categorical model:
  - Objects ≈ (value types, states)
  - Morphisms ≈ functions (arrows) with inferred domains/codomains from runtime sampling
• Runs coherence probes inspired by the Grothendieck–Dirac–MC view:
  - Idempotence check: f(f(x)) == f(x)
  - Commutativity check (binary ops): f(x, y) == f(y, x)
  - Associativity check (binary ops): f(f(x,y), z) == f(x, f(y,z))
  - Functorial (pipeline) check: f∘g vs g∘f where domains line up
  - Stability / sensitivity: small input perturbations shouldn't wildly diverge if coherent
• Computes output entropy and sensitivity to identify high‑entropy, low‑coherence zones
• Emits a structured report (JSON) and a human‑readable summary with ranked "ghost likelihood."

Usage
-----
python ccl.py <path> [--samples 200] [--seed 13] [--report report.json] [--serve]

If --serve is provided, launches a minimal FastAPI server exposing /analyze and /healthz.

Notes
-----
• This is intentionally self‑contained (single file) and avoids heavy deps.
• Runtime sampling executes target functions — run in a sandbox if the code is untrusted.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import io
import json
import math
import os
import pkgutil
import random
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ------------------------------
# Utilities
# ------------------------------

def _shannon_entropy(values: List[Any]) -> float:
    """Compute Shannon entropy (base 2) over a sequence by hashing values to strings."""
    if not values:
        return 0.0
    # Stable string map
    strs = [repr(v) for v in values]
    total = len(strs)
    counts: Dict[str, int] = {}
    for s in strs:
        counts[s] = counts.get(s, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def _numeric_close(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return False


def _deep_equal(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        # allow numeric coercion
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _numeric_close(a, b)
        return False
    if isinstance(a, (int, float)):
        return _numeric_close(a, b)
    if isinstance(a, (str, bytes, bool, type(None))):
        return a == b
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_equal(a[k], b[k]) for k in a.keys())
    # Fallback
    return repr(a) == repr(b)


# ------------------------------
# Sampling inputs
# ------------------------------

PRIMITIVES = (int, float, str, bool)


def _rand_int():
    return random.randint(-1000, 1000)


def _rand_float():
    return random.uniform(-1000.0, 1000.0)


def _rand_str():
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(letters) for _ in range(random.randint(0, 12)))


def _rand_bool():
    return random.choice([True, False])


GENS: Dict[type, Callable[[], Any]] = {int: _rand_int, float: _rand_float, str: _rand_str, bool: _rand_bool}


def _gen_value(depth: int = 0) -> Any:
    """Generate a random value of simple types with shallow containers."""
    base = random.choice([int, float, str, bool])
    v = GENS[base]()
    if depth > 1:
        return v
    # occasionally wrap in containers
    r = random.random()
    if r < 0.2:
        return [GENS[random.choice(list(GENS))]() for _ in range(random.randint(0, 5))]
    if r < 0.3:
        return (GENS[random.choice(list(GENS))]() for _ in range(random.randint(0, 5)))  # generator
    if r < 0.45:
        return (GENS[random.choice(list(GENS))]() , GENS[random.choice(list(GENS))]())
    if r < 0.6:
        return {str(i): GENS[random.choice(list(GENS))]() for i in range(random.randint(0, 4))}
    return v


# ------------------------------
# AST analysis for purity and side-effects heuristics
# ------------------------------

SIDE_EFFECT_CALLS = {
    "print",
    "open",
    "write",
    "writelines",
    "flush",
    "remove",
    "unlink",
    "system",
    "popen",
    "exec",
    "eval",
    "seed",
}

IMPURE_MODULES = {"random", "time", "os", "sys", "subprocess", "socket", "requests"}


def infer_impurity(fn: Callable) -> Dict[str, Any]:
    try:
        src = inspect.getsource(fn)
    except OSError:
        return {"could_inspect": False, "impure": True, "reasons": ["no_source"]}
    try:
        tree = ast.parse(src)
    except Exception:
        return {"could_inspect": False, "impure": True, "reasons": ["parse_failed"]}

    impure = False
    reasons: List[str] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            nonlocal impure
            # detect calls like module.func or bare func
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                # module attribute
                mod = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                if mod in IMPURE_MODULES:
                    impure = True
                    reasons.append(f"module_call:{mod}")
            if name in SIDE_EFFECT_CALLS:
                impure = True
                reasons.append(f"call:{name}")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            # env, stdout etc
            if isinstance(node.value, ast.Name) and node.value.id in {"os", "sys"}:
                nonlocal impure
                impure = True
                reasons.append(f"attr:{node.value.id}.{node.attr}")
            self.generic_visit(node)

        def visit_Global(self, node: ast.Global):
            nonlocal impure
            impure = True
            reasons.append("global")

        def visit_Nonlocal(self, node: ast.Nonlocal):
            nonlocal impure
            impure = True
            reasons.append("nonlocal")

    V().visit(tree)
    return {"could_inspect": True, "impure": impure, "reasons": reasons}


# ------------------------------
# Dynamic probing
# ------------------------------

@dataclass
class ProbeResult:
    idempotent_rate: float
    commutative_rate: Optional[float]
    associative_rate: Optional[float]
    entropy_bits: float
    sensitivity: float
    sample_count: int
    anomalies: List[str] = field(default_factory=list)


@dataclass
class FunctionReport:
    qualname: str
    arity: int
    purity: Dict[str, Any]
    probe: ProbeResult


@dataclass
class PipelineReport:
    pair: Tuple[str, str]
    agreement_rate: float
    samples: int


@dataclass
class AnalysisReport:
    functions: List[FunctionReport]
    pipelines: List[PipelineReport]
    ghost_hotspots: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _call_safely(fn: Callable, *args, **kwargs) -> Tuple[bool, Any]:
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, f"<error:{type(e).__name__}:{e}>"


def _infer_arity(fn: Callable) -> int:
    try:
        sig = inspect.signature(fn)
        ar = 0
        for p in sig.parameters.values():
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                if p.default is inspect._empty:
                    ar += 1
        return ar
    except Exception:
        return 1


def probe_function(fn: Callable, samples: int, seed: int) -> ProbeResult:
    random.seed(seed)
    arity = _infer_arity(fn)
    idemp_hits = 0
    idemp_total = 0
    comm_hits = 0
    comm_total = 0
    assoc_hits = 0
    assoc_total = 0
    outputs: List[Any] = []
    sens_vals: List[float] = []
    anomalies: List[str] = []

    for _ in range(samples):
        if arity == 0:
            ok, y1 = _call_safely(fn)
            if not ok:
                anomalies.append("raise:0-ary")
                continue
            ok2, y2 = _call_safely(lambda: fn())
            if ok2 and _deep_equal(y1, y2):
                idemp_hits += 1
            idemp_total += 1
            outputs.append(y1)
        elif arity == 1:
            x = _gen_value()
            ok, y = _call_safely(fn, x)
            if not ok:
                anomalies.append("raise:unary")
                continue
            ok2, yy = _call_safely(fn, y)
            if ok2 and _deep_equal(y, yy):
                idemp_hits += 1
            idemp_total += 1
            outputs.append(y)
            # sensitivity: small perturbation for numeric x
            if isinstance(x, (int, float)):
                dx = (abs(float(x)) + 1.0) * 1e-6
                x2 = float(x) + dx
                ok3, y2 = _call_safely(fn, x2)
                if ok3:
                    try:
                        diff = abs(float(y2) - float(y))
                        sens_vals.append(diff / (abs(float(y)) + 1e-9))
                    except Exception:
                        pass
        else:  # arity >= 2, treat as binary for checks
            x = _gen_value()
            y = _gen_value()
            ok, r1 = _call_safely(fn, x, y)
            if not ok:
                anomalies.append("raise:binary")
                continue
            # commutativity
            ok2, r2 = _call_safely(fn, y, x)
            if ok2:
                if _deep_equal(r1, r2):
                    comm_hits += 1
                comm_total += 1
            # associativity : (x∘y)∘z vs x∘(y∘z) when output can feed back in
            z = _gen_value()
            ok3, xy = _call_safely(fn, x, y)
            ok4, yz = _call_safely(fn, y, z)
            if ok3 and ok4:
                ok5, a = _call_safely(fn, xy, z)
                ok6, b = _call_safely(fn, x, yz)
                if ok5 and ok6:
                    if _deep_equal(a, b):
                        assoc_hits += 1
                    assoc_total += 1
            # idempotence (binary): f(x,x) == x or == f(x,x) again — weak notion
            ok7, xx = _call_safely(fn, x, x)
            ok8, xxx = _call_safely(fn, xx, xx) if ok7 else (False, None)
            if ok7 and ok8 and _deep_equal(xx, xxx):
                idemp_hits += 1
            idemp_total += 1
            outputs.append(r1)

    ent = _shannon_entropy(outputs)
    sensitivity = sum(sens_vals) / len(sens_vals) if sens_vals else 0.0

    return ProbeResult(
        idempotent_rate=(idemp_hits / idemp_total) if idemp_total else 0.0,
        commutative_rate=(comm_hits / comm_total) if comm_total else None,
        associative_rate=(assoc_hits / assoc_total) if assoc_total else None,
        entropy_bits=ent,
        sensitivity=sensitivity,
        sample_count=idemp_total,
        anomalies=anomalies,
    )


def discover_functions(mod: types.ModuleType) -> List[Tuple[str, Callable]]:
    funcs: List[Tuple[str, Callable]] = []
    for name, obj in vars(mod).items():
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            # skip "dunder" and private helpers by default
            if name.startswith("_"):
                continue
            funcs.append((f"{mod.__name__}.{name}", obj))
    return funcs


def load_modules_from_path(path: Path) -> List[types.ModuleType]:
    modules: List[types.ModuleType] = []
    path = path.resolve()
    if path.is_file() and path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)  # type: ignore
            modules.append(mod)
        return modules

    # Add directory to sys.path for package style imports
    sys.path.insert(0, str(path))
    for finder, name, ispkg in pkgutil.iter_modules([str(path)]):
        try:
            mod = importlib.import_module(name)
            modules.append(mod)
        except Exception:
            continue
    return modules


def analyze_path(target: Union[str, Path], samples: int, seed: int) -> AnalysisReport:
    random.seed(seed)
    target_path = Path(target)
    modules = load_modules_from_path(target_path)
    all_funcs: List[Tuple[str, Callable]] = []
    for m in modules:
        all_funcs.extend(discover_functions(m))

    fn_reports: List[FunctionReport] = []
    for qual, fn in all_funcs:
        purity = infer_impurity(fn)
        probe = probe_function(fn, samples=samples, seed=seed)
        fn_reports.append(
            FunctionReport(
                qualname=qual,
                arity=max(0, _infer_arity(fn)),
                purity=purity,
                probe=probe,
            )
        )

    # Pipeline coherence: try f∘g vs g∘f when arities allow unary
    unaries = [(q, f) for (q, f) in all_funcs if max(0, _infer_arity(f)) == 1]
    pipe_reports: List[PipelineReport] = []
    for i in range(len(unaries)):
        for j in range(i + 1, len(unaries)):
            (qf, f), (qg, g) = unaries[i], unaries[j]
            agree = 0
            total = 0
            for _ in range(max(10, samples // 5)):
                x = _gen_value()
                ok1, a = _call_safely(lambda v: f(g(v)), x)
                ok2, b = _call_safely(lambda v: g(f(v)), x)
                if ok1 and ok2:
                    total += 1
                    if _deep_equal(a, b):
                        agree += 1
            if total:
                pipe_reports.append(
                    PipelineReport(pair=(qf, qg), agreement_rate=agree / total, samples=total)
                )

    # Rank ghost hotspots: high entropy + low idempotence + high sensitivity or many anomalies
    scored: List[Tuple[float, FunctionReport]] = []
    for fr in fn_reports:
        p = fr.probe
        score = (
            (p.entropy_bits) * 0.5
            + (1.0 - p.idempotent_rate) * 0.3
            + (min(1.0, p.sensitivity)) * 0.2
            + (len(p.anomalies) > 0) * 0.1
            + (0.1 if fr.purity.get("impure") else 0.0)
        )
        scored.append((float(score), fr))
    scored.sort(key=lambda t: t[0], reverse=True)

    hotspots = [
        {
            "function": fr.qualname,
            "score": round(score, 4),
            "entropy_bits": round(fr.probe.entropy_bits, 4),
            "idempotent_rate": round(fr.probe.idempotent_rate, 4),
            "sensitivity": round(fr.probe.sensitivity, 6),
            "anomalies": fr.probe.anomalies,
            "impure": fr.purity.get("impure", True),
            "reasons": fr.purity.get("reasons", []),
        }
        for score, fr in scored[:10]
    ]

    summary = {
        "modules": [m.__name__ for m in modules],
        "functions_analyzed": len(fn_reports),
        "pipelines_checked": len(pipe_reports),
        "top_hotspots": hotspots[:5],
    }

    return AnalysisReport(functions=fn_reports, pipelines=pipe_reports, ghost_hotspots=hotspots, summary=summary)


def report_to_json(ar: AnalysisReport) -> Dict[str, Any]:
    return {
        "summary": ar.summary,
        "functions": [
            {
                "qualname": f.qualname,
                "arity": f.arity,
                "purity": f.purity,
                "probe": {
                    "idempotent_rate": f.probe.idempotent_rate,
                    "commutative_rate": f.probe.commutative_rate,
                    "associative_rate": f.probe.associative_rate,
                    "entropy_bits": f.probe.entropy_bits,
                    "sensitivity": f.probe.sensitivity,
                    "sample_count": f.probe.sample_count,
                    "anomalies": f.probe.anomalies,
                },
            }
            for f in ar.functions
        ],
        "pipelines": [
            {
                "pair": pr.pair,
                "agreement_rate": pr.agreement_rate,
                "samples": pr.samples,
            }
            for pr in ar.pipelines
        ],
        "ghost_hotspots": ar.ghost_hotspots,
    }


def print_human_summary(ar: AnalysisReport) -> None:
    s = ar.summary
    print("\n=== Categorical Coherence Linter (CCL) ===")
    print(f"Analyzed modules: {', '.join(s['modules'])}")
    print(f"Functions analyzed: {s['functions_analyzed']} | Pipelines checked: {s['pipelines_checked']}")
    print("\nTop ghost hotspots:")
    if not s["top_hotspots"]:
        print("  (none)")
    for i, h in enumerate(s["top_hotspots"], 1):
        print(
            f" {i}. {h['function']}  score={h['score']}  H={h['entropy_bits']}  "
            f"idemp={h['idempotent_rate']}  sens={h['sensitivity']}  impure={h['impure']}"
        )
        if h.get("anomalies"):
            print(f"    anomalies: {h['anomalies']}")
        if h.get("reasons"):
            print(f"    impurity-reasons: {h['reasons']}")

    if ar.pipelines:
        print("\nPipeline (f∘g vs g∘f) agreement (lower can indicate hidden non-commutativity):")
        pl = sorted(ar.pipelines, key=lambda p: p.agreement_rate)
        for p in pl[:10]:
            print(f"  {p.pair[0]} ∘ {p.pair[1]} vs {p.pair[1]} ∘ {p.pair[0]}  -> agree={p.pair[0]} ∘ {p.pair[1]} vs {p.pair[1]} ∘ {p.pair[0]}  -> agree={p.agreement_rate:.3f} (n={p.samples})")


# ------------------------------
# Optional FastAPI server
# ------------------------------

def maybe_serve():
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except Exception:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(2)

    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ts": time.time()}

    @app.post("/analyze")
    def analyze(payload: Dict[str, Any]):
        target = payload.get("path")
        samples = int(payload.get("samples", 200))
        seed = int(payload.get("seed", 13))
        if not target:
            return JSONResponse({"error": "path required"}, status_code=400)
        ar = analyze_path(target, samples=samples, seed=seed)
        return report_to_json(ar)

    uvicorn.run(app, host="0.0.0.0", port=8080)


# ------------------------------
# CLI
# ------------------------------


def main():
    ap = argparse.ArgumentParser(description="Categorical Coherence Linter (entropy-driven)")
    ap.add_argument("path", help="Path to a .py file or a package directory to analyze")
    ap.add_argument("--samples", type=int, default=200, help="Number of random samples per probe")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed (reproducible)")
    ap.add_argument("--report", type=str, default=None, help="Write full JSON report to this file")
    ap.add_argument("--serve", action="store_true", help="Start a FastAPI server instead of CLI analysis")
    args = ap.parse_args()

    if args.serve:
        maybe_serve()
        return

    ar = analyze_path(args.path, samples=args.samples, seed=args.seed)
    print_human_summary(ar)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_to_json(ar), f, indent=2)
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main() 