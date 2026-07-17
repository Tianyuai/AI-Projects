"""Safe server-rendered collaborator UI for auditable search results."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Protocol
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from paper_search.domain.models import Paper
    from paper_search.evaluation.runner import PipelineResult


_INVALID_QUERY_MESSAGE = "query must be provided exactly once"
_UNAVAILABLE_MESSAGE = "search is temporarily unavailable"


class _InvalidQuery(ValueError):
    """The URL-encoded request did not contain one non-empty query."""


class _SearchUnavailable(RuntimeError):
    """The application composition layer has not supplied a search service."""


class SearchService(Protocol):
    """Injected asynchronous boundary for the search pipeline."""

    async def __call__(self, query: str) -> PipelineResult: ...


def _escaped(value: str) -> str:
    return escape(value, quote=True)


def _render_form() -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Paper search</title></head><body>"
        "<main><h1>Paper search</h1>"
        '<form method="post" action="/search">'
        '<label for="query">Research query</label>'
        '<input id="query" name="query" type="search" required>'
        '<button type="submit">Search</button>'
        "</form></main></body></html>"
    )


def _paper_metadata(paper: Paper) -> str:
    authors = ", ".join(_escaped(author) for author in paper.authors)
    year = (
        _escaped(str(paper.publication_year))
        if paper.publication_year is not None
        else "Unknown"
    )
    venue = _escaped(paper.venue) if paper.venue is not None else "Unknown"
    sources = ", ".join(_escaped(source) for source in paper.sources) or "Unknown"
    return (
        f"<p>Authors: {authors or 'Unknown'}</p>"
        f"<p>Year: {year}</p>"
        f"<p>Venue: {venue}</p>"
        f"<p>Sources: {sources}</p>"
    )


def _render_results(query: str, result: PipelineResult) -> str:
    ranked_items = "".join(
        "<li>"
        f"<h2>{_escaped(score.paper.title)}</h2>"
        f"{_paper_metadata(score.paper)}"
        f"<p>Final score: {_escaped(str(score.final_score))}</p>"
        "</li>"
        for score in result.ranked
    )
    merge_items = "".join(
        "<li>"
        f"Representative: {_escaped(decision.representative_id)}; "
        f"members: {_escaped(', '.join(decision.member_ids))}; "
        f"match rule: {_escaped(decision.match_rule)}; "
        f"match value: {_escaped(decision.match_value)}"
        "</li>"
        for decision in result.deduplication.decisions
    )
    rejected_items = "".join(
        "<li>"
        f"{_escaped(item.paper.title)}: {_escaped(item.reason_code)} "
        f"({_escaped(item.reason)})"
        "</li>"
        for item in result.filtering.rejected
    )
    uncertainty_items = "".join(
        "<li>"
        f"{_escaped(item.paper.title)}: "
        f"{_escaped(', '.join(item.uncertainty_reasons)) or 'None'}"
        "</li>"
        for item in result.filtering.accepted
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Paper search results</title></head><body><main>"
        f"<h1>Results for {_escaped(query)}</h1>"
        f"<ol>{ranked_items}</ol>"
        f"<details><summary>Merge decisions</summary><ul>{merge_items}</ul></details>"
        f"<details><summary>Rejected papers</summary><ul>{rejected_items}</ul></details>"
        "<details><summary>Accepted-paper uncertainty reasons</summary>"
        f"<ul>{uncertainty_items}</ul></details>"
        "</main></body></html>"
    )


def _parse_query(body: bytes) -> str:
    try:
        fields = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError as error:
        raise _InvalidQuery from error
    values = fields.get("query", [])
    if len(values) != 1 or not values[0].strip():
        raise _InvalidQuery
    return values[0].strip()


def create_app(search_service: SearchService) -> FastAPI:
    """Create an ASGI application with an injected search service."""
    application = FastAPI()

    @application.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return _render_form()

    @application.post("/search", response_class=HTMLResponse)
    async def search(request: Request) -> HTMLResponse:
        try:
            query = _parse_query(await request.body())
        except _InvalidQuery:
            return HTMLResponse(_INVALID_QUERY_MESSAGE, status_code=400)
        try:
            result = await search_service(query)
        except Exception:
            return HTMLResponse(_UNAVAILABLE_MESSAGE, status_code=503)
        return HTMLResponse(_render_results(query, result))

    return application


async def _unavailable_service(query: str) -> PipelineResult:
    del query
    raise _SearchUnavailable


app = create_app(_unavailable_service)
