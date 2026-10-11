"""FastAPI inspection API. Run with: uvicorn api.main:app --reload"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dashboard import router as data_router
from api.routes import router

app = FastAPI(title="TrustyShelf inspection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(data_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
