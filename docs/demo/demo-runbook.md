# Demonstration Runbook

## Preconditions

Run these commands from the repository root with the project's normal, prepared `uv` environment already available (including the installed package and its core dependencies). This runbook intentionally does not provision that environment, because provisioning may require dependency installation. It uses only the fixed, loopback-only synthetic mock service and does not require provider credentials, a real provider, a real LLM, a dataset, or an external network connection. Keep the `--no-env-file` flag on every `uv` command so the command does not load a `.env` file.

Use two PowerShell terminals: start the service in the first and issue the requests in the second.

## Start the Mock API

In the first terminal, start the mock entry point on its fixed loopback host:

```powershell
D:\Dev\uv\uv.exe run --no-sync --no-env-file python -m paper_search.api.mock_server --host 127.0.0.1 --port 8000
```

Leave this terminal running. The mock server installs a loopback-only network guard and composes the fixed synthetic search service.

## Health Check

In the second terminal, check liveness and then readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Liveness shows that the process is serving. Readiness checks the injected synthetic service and its fixed mock provider-status map.

## Run a Demonstration Query

Submit a request with the trace enabled:

```powershell
$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/search `
  -ContentType 'application/json' `
  -Body '{"query_id":"demo-q1","query":"graph retrieval","budget_profile":"balanced","include_trace":true}'

$response | ConvertTo-Json -Depth 10
```

The request body matches the mock API contract. The response is synthetic evidence of this composition only; it is not a provider-backed search result.

## Inspect Trace and Usage

Inspect the response object from the preceding command without inferring a quality result:

```powershell
$response.search_trace | ConvertTo-Json -Depth 10
$response.usage | ConvertTo-Json -Depth 10
$response.stop_reason
$response.is_partial
$response.warnings
```

The trace documents the stages executed by this request. Usage is the mock orchestrator's committed accounting for the request; it is not a measured cost or timing report.

## Demonstrate Provider Degradation

Do not simulate degradation by taking down, blocking, or contacting a real provider. Instead, run the following isolated fake-readiness demonstration. It reuses the mock server's synthetic composition and injects a false value into its readiness map; the ASGI transport keeps the check in-process rather than making a network call.

```powershell
@'
import asyncio
from unittest.mock import patch

import httpx

from paper_search.api import mock_server


async def main() -> None:
    with patch.object(
        mock_server,
        "mock_readiness",
        return_value={"openalex": True, "semantic_scholar": False},
    ):
        app = mock_server.create_mock_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
            response = await client.get("/health/ready")
    print(response.status_code)
    print(response.json())


asyncio.run(main())
'@ | D:\Dev\uv\uv.exe run --no-sync --no-env-file python -
```

This demonstrates the documented readiness behavior for an injected fake: a non-ready provider makes `/health/ready` report degraded. It does not modify the running mock server in the first terminal.

## Stop the Service

Return to the first terminal and press `Ctrl+C`. The mock server then stops; no provider shutdown action is needed because the service has no real provider connections.

## Outputs Deferred Until R3

This runbook intentionally records no formal metrics, costs, screenshots, or timing measurements. R2 is retrieval diagnostic evidence only, and no relevance metrics are presented here. R3 is the later boundary for formal evaluation artifacts and any measured conclusions.
