from fastapi import APIRouter, Body, HTTPException
from ..db import create_db_and_tables
from ..tools.ccl import analyze_path, report_to_json
from pathlib import Path
import tempfile
import os

router = APIRouter(tags=["tools"], prefix="/tools")

@router.post("/init-db")
def initialize_database():
    """Initialize database tables"""
    try:
        create_db_and_tables()
        return {"ok": True, "message": "Database initialized successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.get("/status")
def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "service": "carryon-mvp",
        "features": ["memories", "soulpacks", "embeddings", "tone-alignment", "vector-search", "ccl"]
    }

@router.post("/ccl-analyze")
def ccl_analyze(payload: dict = Body(...)):
    """Run Categorical Coherence Linter analysis on specified path"""
    try:
        target_path = payload.get("path", "")
        samples = int(payload.get("samples", 100))
        seed = int(payload.get("seed", 42))
        
        if not target_path:
            raise HTTPException(400, "path is required")
        
        # Validate path exists and is within project bounds
        target = Path(target_path)
        if not target.exists():
            raise HTTPException(404, f"Path {target_path} not found")
        
        # Security: ensure path is within project directory
        project_root = Path(__file__).parent.parent.parent
        try:
            target = target.resolve()
            project_root = project_root.resolve()
            if not str(target).startswith(str(project_root)):
                raise HTTPException(400, "Path must be within project directory")
        except Exception:
            raise HTTPException(400, "Invalid path")
        
        # Run CCL analysis
        analysis = analyze_path(target, samples=samples, seed=seed)
        report = report_to_json(analysis)
        
        return {
            "ok": True,
            "analysis": report,
            "target": str(target),
            "samples": samples,
            "seed": seed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"CCL analysis failed: {str(e)}"}

@router.post("/ccl-quick")
def ccl_quick_scan():
    """Run quick CCL scan on core project files"""
    try:
        project_root = Path(__file__).parent.parent.parent
        core_files = [
            project_root / "server" / "app" / "retrieval" / "vector_index.py",
            project_root / "server" / "app" / "tone" / "alignment.py",
            project_root / "server" / "app" / "retrieval" / "ranker.py"
        ]
        
        results = []
        for file_path in core_files:
            if file_path.exists():
                try:
                    analysis = analyze_path(file_path, samples=50, seed=42)
                    results.append({
                        "file": str(file_path.name),
                        "functions_analyzed": analysis.summary["functions_analyzed"],
                        "top_hotspots": analysis.summary["top_hotspots"][:3] if analysis.summary["top_hotspots"] else []
                    })
                except Exception as e:
                    results.append({
                        "file": str(file_path.name),
                        "error": str(e)
                    })
        
        return {
            "ok": True,
            "quick_scan": results,
            "message": "Quick CCL scan completed"
        }
        
    except Exception as e:
        return {"ok": False, "error": f"Quick CCL scan failed: {str(e)}"} 