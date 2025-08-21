from fastapi import APIRouter

router = APIRouter(tags=["health"], prefix="/health")

@router.get("/")
def health_check():
    return {"status": "healthy", "service": "carryon-mvp"} 