from __future__ import annotations

import importlib

import pytest
import spanlight

from app.main import SERVICE_NAME, create_app


def test_the_chassis_bootstrap_is_gone() -> None:
    """A test rather than a comment, because a fork is how it came back before.

    `app/otel_bootstrap.py` passed the endpoint to the exporter unchanged, so spans
    went to a path that rejects them, and it never percent-decoded the auth header,
    so the request 401'd. It also called `set_tracer_provider`, which OpenTelemetry
    ignores on a second call with only a log line, so running it beside
    `spanlight.init` made the winner a coin toss. Three sibling repos carried this
    and none had ever exported a span to Grafana. This repo is the fourth fork of
    that chassis, which is exactly why the check is executable.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.otel_bootstrap")


def test_tracing_is_off_when_no_endpoint_is_configured() -> None:
    """Pins Spanlight's guarantee, not ours, which is worth a test anyway.

    The whole suite calls `create_app`, so if this returned true without an endpoint
    every test run would build an exporter and a metric reader posting to a dead
    host. Spanlight's own repo lost four and a half minutes of suite time to exactly
    that.
    """
    assert spanlight.init(SERVICE_NAME, endpoint=None, headers=None) is False


def test_create_app_survives_an_empty_endpoint_string() -> None:
    """`.env.example` ships `OTEL_EXPORTER_OTLP_ENDPOINT=` with no value, so the
    setting arrives as an empty string rather than as None. Falsy either way, and
    the point is that nothing downstream treats the empty string as a URL."""
    assert create_app().title == SERVICE_NAME
