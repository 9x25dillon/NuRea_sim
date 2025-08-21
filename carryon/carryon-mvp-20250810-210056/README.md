# CarryOn – Portable AI Identity (MVP)

Keep your AI *you* across model updates, app wipes, and devices.

## Quick Start (Backend)
```bash
make install
make dev
# open http://localhost:8000/docs
```

## Docker
```bash
cd ops
docker compose up --build
```

## Structure
- `server/` FastAPI API
- `apps/desktop/` React (Vite) UI stubs
- `ops/` Docker Compose
- `examples/` Sample soulpack + memories
