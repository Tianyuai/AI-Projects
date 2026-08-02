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
from paper_search.api.routing import SearchServiceRouter
from paper_search.application.contracts import ReadyHealthResponse
from paper_search.domain.models import DependencyStatus
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service


_LOOPBACK_HOST = "127.0.0.1"
_NETWORK_ERROR = "mock server blocks non-loopback network access"
_SOCKET_TARGET_EVENTS = frozenset(
    {
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.sendmsg",
        "socket.sendto",
    }
)
_NAME_TARGET_EVENTS = frozenset(
    {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
    }
)


def mock_readiness() -> dict[str, bool]:
    """Return the fixed readiness map for the offline mock composition."""
    return {
        "openalex": True,
        "semantic_scholar": True,
    }


def create_mock_app() -> FastAPI:
    """Build the only app composition exposed by the mock-server entry point."""
    router = SearchServiceRouter(
        replay_service=build_synthetic_search_service(),
        readiness=ReadyHealthResponse(
            status="ready",
            execution_mode="replay",
            snapshot_set_id="mock-snapshot-v1",
            dependencies=[
                DependencyStatus(
                    dependency=dependency,
                    state="ready",
                    cache_hit=False,
                    error_codes=[],
                )
                for dependency in ("llm", "openalex", "semantic_scholar")
            ],
            last_authorized_probe_at=None,
        ),
    )
    return create_app(router)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _loopback_host(value: str) -> str:
    if value != _LOOPBACK_HOST:
        raise argparse.ArgumentTypeError("host must be 127.0.0.1")
    return value


def _is_loopback_name(value: object) -> bool:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(value, str):
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _is_loopback_target(value: object) -> bool:
    if not isinstance(value, tuple) or not value or not isinstance(value[0], str):
        return False
    return _is_loopback_name(value[0])


def _socket_target(args: tuple[Any, ...]) -> object:
    for candidate in reversed(args[1:]):
        if isinstance(candidate, tuple):
            return candidate
    return None


def _audit_network(event: str, args: tuple[Any, ...]) -> None:
    if event in _SOCKET_TARGET_EVENTS and not _is_loopback_target(_socket_target(args)):
        raise RuntimeError(_NETWORK_ERROR)
    if event == "socket.getnameinfo":
        if not args or not _is_loopback_target(args[0]):
            raise RuntimeError(_NETWORK_ERROR)
        return
    if event in _NAME_TARGET_EVENTS and args and not _is_loopback_name(args[0]):
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
    try:
        uvicorn.run(
            create_mock_app(),
            host=args.host,
            port=args.port,
            access_log=False,
            log_level="warning",
        )
    except OSError:
        print("mock server startup failed", file=sys.stderr)
        return 2
    except SystemExit as error:
        if error.code not in (None, 0):
            print("mock server startup failed", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
