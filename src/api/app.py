from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    get_rag_pipeline,
    shutdown_rag_pipeline,
)
from src.api.routes.ask import (
    router as ask_router,
)
from src.api.routes.config import (
    router as config_router,
)
from src.api.routes.health import (
    router as health_router,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Cycle de vie de l'application.

    Au démarrage :
    - initialise le pipeline RAG une seule fois.

    À l'arrêt :
    - libère les modèles ;
    - nettoie le cache.
    """

    print("=" * 80)
    print("DÉMARRAGE DE L'API RAG")
    print("=" * 80)

    pipeline = get_rag_pipeline()

    print(
        "Provider :",
        pipeline.llm_manager.provider_name,
    )

    print(
        "Model :",
        pipeline.llm_manager.model_name,
    )

    print(
        "Generation contexts :",
        pipeline.generation_contexts,
    )

    print("API prête.")

    try:
        yield

    finally:
        print()
        print("=" * 80)
        print("ARRÊT DE L'API RAG")
        print("=" * 80)

        shutdown_rag_pipeline()

        print(
            "Pipeline libéré."
        )


app = FastAPI(
    title="Internal Knowledge Assistant API",
    description=(
        "API REST exposant le pipeline RAG "
        "de l'assistant documentaire interne."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routes
# ============================================================

app.include_router(
    health_router
)

app.include_router(
    config_router
)

app.include_router(
    ask_router
)


# ============================================================
# Route racine
# ============================================================

@app.get(
    "/",
    tags=["root"],
)
def root() -> dict[str, str]:
    """
    Route minimale permettant de vérifier
    que FastAPI fonctionne.
    """

    return {
        "service": (
            "Internal Knowledge Assistant API"
        ),
        "status": "running",
        "documentation": "/docs",
    }