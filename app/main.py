from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .db import connect, disconnect


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()  # create pool + initialise schema
    yield
    await disconnect()


app = FastAPI(
    title="SEC Filings Research Assistant",
    description=(
        "Agentic RAG over SEC EDGAR filings. "
        "Informational only — not investment advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Dev-friendly CORS so the Next.js frontend (localhost:3000) can call the API.
# Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
