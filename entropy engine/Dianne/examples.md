# Chaos LLM Julia Integration Examples

## Async Suggest with WebSocket Evaluation

This request uses the async suggest endpoint with WebSocket-based evaluation:

```bash
curl -sX POST localhost:8000/suggest \
  -H 'content-type: application/json' \
  -d '{"prefix":"VAR(1,2,3)","state":"S0","use_semantic":true,"async_eval":true}' | jq