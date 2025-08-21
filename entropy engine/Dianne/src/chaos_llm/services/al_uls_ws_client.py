import os
import asyncio
import websockets
import json
from typing import Dict, Any, List

JULIA_WS_URL = os.environ.get("JULIA_WS_URL", "ws://localhost:8089")

class ALULSWSClient:
    def __init__(self, ws_url: str | None = None):
        self.ws_url = ws_url or JULIA_WS_URL
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_cache_key(self, name: str, args: List[str]) -> str:
        return f"{name}:{'|'.join(args)}"

    async def connect(self):
        if (self.websocket is None) or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)
        return self.websocket

    async def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            return json.loads(resp)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        cache_key = f"parse:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self._roundtrip({"type": "parse", "text": text})
        self._cache[cache_key] = result
        return result

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cache_key = self._make_cache_key(name, args)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = await self._roundtrip({"type": "eval", "name": name, "args": args})
        self._cache[cache_key] = result
        return result

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
        
        # Evaluate uncached calls via WebSocket
        if uncached_calls:
            res = await self._roundtrip({"type": "batch_eval", "calls": uncached_calls})
            results = res.get("results", []) if isinstance(res, dict) else [{"ok": False, "error": "invalid response"}]
            
            # Update cache
            for idx, result in zip(uncached_indices, results):
                cache_key = self._make_cache_key(uncached_calls[idx].get("name", ""), uncached_calls[idx].get("args", []))
                self._cache[cache_key] = result
        else:
            results = []
        
        # Reconstruct full results list
        final_results = [None] * len(calls)
        for i, result in cached_results:
            final_results[i] = result
        for i, result in zip(uncached_indices, results):
            final_results[i] = result
            
        return [r for r in final_results if r is not None]

al_uls_ws_client = ALULSWSClient()