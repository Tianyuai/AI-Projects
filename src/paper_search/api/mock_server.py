"""Loopback-only offline server for the fixed Week 2 synthetic mock stack."""

from __future__ import annotations

import argparse
import ipaddress
import sys
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
import uvicorn

from paper_search.api.app import create_app
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service


_LOOPBACK_HOST = "127.0.0.1"
_NETWORK_ERROR = "mock server blocks non-loopback network access"


def mock_readiness() -> dict[str, bool]:
    """Return the fixed readiness map for the offline mock composition."""
    return {
        "openalex": True,
        "semantic_scholar": True,
    }


def create_mock_app() -> FastAPI:
    """Build the only app composition exposed by the mock-server entry point."""
    return create_app(
        build_synthetic_search_service(),
        readiness_probe=mock_readiness,
    )


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _loopback_host(value: str) -> str:
    if value != _LOOPBACK_HOST:
        raise argparse.ArgumentTypeError("host must be 127.0.0.1")
    return value


def _is_loopback_target(value: object) -> bool:
    if not isinstance(value, tuple) or not value or not isinstance(value[0], str):
        return False
    try:
        return ipaddress.ip_address(value[0]).is_loopback
    except ValueError:
        return False


def _audit_network(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect" and len(args) == 2 and not _is_loopback_target(args[1]):
        raise RuntimeError(_NETWORK_ERROR)
    if event == "socket.getaddrinfo" and args and args[0] not in {None, _LOOPBACK_HOST}:
        raise RuntimeError(_NETWORK_ERROR)


def _install_loopback_only_guard() -> None:
    sys.addaudithook(_audit_network)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed offline Week 2 mock API",
        allow_abbrev=False,
    )
    parser.add_argument("--host", type=_loopback_host, default=_LOOPBACK_HOST)
    parser.add_argument("--port", type=_port, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _install_loopback_only_guard()
    uvicorn.run(
        create_mock_app(),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
