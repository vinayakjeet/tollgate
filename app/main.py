from __future__ import annotations

import spanlight
from fastapi import FastAPI

from app.auth import maybe_warn_unauthenticated
from app.config import get_settings, load_dotenv_into_environ
from app.gateway import Gateway
from app.logging_config import configure_logging
from app.metering import JsonlMeteringStore
from app.middleware import RequestContextMiddleware
from app.routers import budget, demo, health, v1

SERVICE_NAME = "tollgate"


def create_app() -> FastAPI:
    load_dotenv_into_environ()
    settings = get_settings()
    configure_logging(settings.log_level)

    # Spanlight owns tracing setup. The chassis `otel_bootstrap` this replaces
    # passed the endpoint straight to the exporter, which appends nothing, so every
    # span went to a URL that does not accept spans. It also never percent-decoded
    # the auth header, earning a 401 that reads like a bad credential. Neither is
    # visible from inside the process: the log said enabled either way. Three
    # sibling repos carried it and none had ever exported a span.
    spanlight.init(
        SERVICE_NAME,
        endpoint=settings.otel_exporter_otlp_endpoint,
        headers=settings.otel_exporter_otlp_headers,
    )

    app = FastAPI(title=SERVICE_NAME, version="0.1.0")
    app.add_middleware(RequestContextMiddleware)

    maybe_warn_unauthenticated()

    # The Postgres-backed metering store arrives with the Neon credentials; until
    # then rows land in a local JSONL file, which restarts survive and scripts can
    # read. A missing database must not mean missing history.
    gateway = Gateway.from_settings(
        settings, metering=JsonlMeteringStore(settings.metering_path)
    )
    app.state.gateway = gateway

    app.include_router(health.router)
    app.include_router(budget.router)
    app.include_router(demo.router)
    app.include_router(v1.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
