from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, soulpacks, memories, prime, tools, embeddings, tone

app = FastAPI(title="CarryOn MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(soulpacks.router, prefix="/v1")
app.include_router(memories.router, prefix="/v1")
app.include_router(prime.router, prefix="/v1")
app.include_router(tools.router, prefix="/v1")
app.include_router(embeddings.router, prefix="/v1")
app.include_router(tone.router, prefix="/v1")

@app.get("/")
def root():
    return {"message": "CarryOn MVP API", "version": "0.1.0"} 