import os
import re
from typing import Dict, Any, Optional, List
from .al_uls_client import al_uls_client
from .al_uls_ws_client import al_uls_ws_client

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$")
USE_WEBSOCKET = bool(os.environ.get("JULIA_WS_URL"))

class ALULS:
    def is_symbolic_call(self, text: str) -> bool:
        return bool(CALL_RE.search((text or "").strip()))

    def parse_symbolic_call(self, text: str) -> Dict[str, Any]:
        m = CALL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split(",") if a.strip()]
        return {"name": name.upper(), "args": args}

    async def eval_symbolic_call_async(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if USE_WEBSOCKET:
                return await al_uls_ws_client.eval(call.get("name", ""), call.get("args", []))
            else:
                return await al_uls_client.eval(call.get("name", ""), call.get("args", []))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval_symbolic_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if USE_WEBSOCKET:
            return await al_uls_ws_client.batch_eval(calls)
        else:
            return await al_uls_client.batch_eval(calls)

al_uls = ALULS()