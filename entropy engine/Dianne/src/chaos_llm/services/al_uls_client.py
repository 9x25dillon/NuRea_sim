import os
import httpx
import asyncio
from typing import Dict, Any, List

JULIA_SERVER_URL = os.environ.get("JULIA_SERVER_URL", "http://localhost:8088")

class ALULSClient:
    def __init__(self, base_url: str | None = None):
        self.base = base_url or JULIA_SERVER_URL
        self.client = httpx.AsyncClient(timeout=10)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_cache_key(self, name: str, args: List[str]) -> str:
        return f"{name}:{'|'.join(args)}"

    async def parse(self, text: str) -> Dict[str, Any]:
        cache_key = f"parse:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/parse", json={"text": text})
            r.raise_for_status()
            result = r.json()
            self._cache[cache_key] = result
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cache_key = self._make_cache_key(name, args)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/eval", json={"name": name, "args": args})
            r.raise_for_status()
            result = r.json()
            self._cache[cache_key] = result
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Check cache for individual calls
        cached_results = []
        uncached_calls = []
        uncached_indices = []
        
        for i, call in enumerate(calls):
            name = call.get("name", "")
            args = call.get("args", [])
            cache_key = self._make_cache_key(name, args)
            if cache_key in self._cache:
                cached_results.append((i, self._cache[cache_key]))
            else:
                uncached_calls.append(call)
                uncached_indices.append(i)
        
        # Evaluate uncached calls
        if uncached_calls:
            tasks = [self.eval(c.get("name", ""), c.get("args", [])) for c in uncached_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out: List[Dict[str, Any]] = []
            for res in results:
                if isinstance(res, Exception):
                    out.append({"ok": False, "error": str(res)})
                else:
                    out.append(res)
            
            # Update cache and merge results
            for idx, result in zip(uncached_indices, out):
                cache_key = self._make_cache_key(calls[idx].get("name", ""), calls[idx].get("args", []))
                self._cache[cache_key] = result
        else:
            out = []
        
        # Reconstruct full results list
        final_results = [None] * len(calls)
        for i, result in cached_results:
            final_results[i] = result
        for i, result in zip(uncached_indices, out):
            final_results[i] = result
            
        return [r for r in final_results if r is not None]

al_uls_client = ALULSClient()