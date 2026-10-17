
"""Wikipedia search tool with a persistent source-of-truth log."""

import re
from html import unescape
from pathlib import Path

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SOURCE_LOG = OUTPUT_DIR / "retrieved_sources.md"

# Store only results returned by the Wikipedia API during this run.
_RETRIEVED_SOURCES: dict[str, dict[str, str]] = {}


class WikipediaSearchInput(BaseModel):
    """Input schema for searching Wikipedia."""

    query: str = Field(
        ...,
        description="The research topic or question to search for.",
    )


def reset_search_log() -> None:
    """Start a fresh source log before the web-enhanced experiment."""
    _RETRIEVED_SOURCES.clear()
    if SOURCE_LOG.exists():
        SOURCE_LOG.unlink()


def get_retrieved_sources() -> list[dict[str, str]]:
    """Return the exact sources captured by the search tool."""
    return list(_RETRIEVED_SOURCES.values())


def format_sources_markdown(
    sources: list[dict[str, str]],
) -> str:
    """Format captured search results without inventing references."""
    lines = ["## Retrieved Wikipedia search results", ""]

    if not sources:
        lines.extend(
            [
                "No results were captured by the Wikipedia search tool.",
                "No retrieved source URLs are claimed for this run.",
            ]
        )
        return "\n".join(lines)

    for source in sources:
        lines.append(f"- **{source['title']}**")
        lines.append(f"  - URL: {source['url']}")
        if source["snippet"]:
            lines.append(f"  - Search snippet: {source['snippet']}")
        lines.append("")

    return "\n".join(lines).rstrip()



def _write_source_log() -> None:
    """Persist tool-returned results so they can be audited."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = [
        "# Wikipedia Search Results Captured During W9D1",
        "",
        (
            "These entries were recorded directly by the search tool. "
            "Their inclusion does not independently verify the accuracy "
            "of claims in the generated report."
        ),
        "",
        format_sources_markdown(get_retrieved_sources()),
        "",
    ]
    SOURCE_LOG.write_text("\n".join(content), encoding="utf-8")


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia and log its actual returned results."""

    name: str = "Wikipedia Web Search"
    description: str = (
        "Search Wikipedia for relevant information. Returns article "
        "titles, search snippets, and the URLs returned by the API."
    )
    args_schema: type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str) -> str:
        """Query the public Wikipedia API and log each returned result."""
        if not query.strip():
            return "Search error: the query cannot be empty."

        endpoint = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        headers = {
            "User-Agent": (
                "CynarisW9D1ResearchCrew/1.0 "
                "(educational project)"
            )
        }

        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
        except (requests.RequestException, ValueError) as exc:
            return f"Web search unavailable: {exc}"

        if not results:
            return f"No Wikipedia articles found for: {query}"

        formatted_results = []

        for index, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            snippet = unescape(snippet).strip()
            page_id = item.get("pageid")

            if not page_id:
                continue

            url = f"https://en.wikipedia.org/?curid={page_id}"
            _RETRIEVED_SOURCES[url] = {
                "title": title,
                "snippet": snippet,
                "url": url,
            }

            formatted_results.append(
                f"{index}. {title}\n"
                f"   Summary: {snippet}\n"
                f"   Source: {url}"
            )

        if not formatted_results:
            return "Wikipedia returned no usable article URLs."

        _write_source_log()
        return "\n\n".join(formatted_results)
    