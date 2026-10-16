# W9D2 Research Quality Comparison

## Experiment setup

- Baseline: crew run without the web-search tool.
- Web-enhanced: crew run with the Wikipedia search tool available.
- Code-execution tool: restricted arithmetic calculator.
- Comparison method: word counts, URL counts, and captured source count.

## Observations

| Measure | Baseline | Web-enhanced |
|---|---:|---:|
| Word count | 431 | 477 |
| URL occurrences | 0 | 5 |
| Captured search sources | Not applicable | 5 |

## Source observation

The search tool recorded 5 source(s). Inspect the saved source list to assess relevance and quality.

## Interpretation

The web-enhanced run can provide traceable search results when the API
succeeds and the researcher uses the tool. URL counts and word counts are
descriptive measurements, not measures of factual accuracy.

A larger report or more URLs does not by itself prove higher quality.
Review both reports for coverage, unsupported claims, relevance, balance,
and whether claims are supported by the captured sources.

## Limitations

- Wikipedia search results are not independently verified evidence.
- Search results can be incomplete, outdated, or irrelevant.
- Local model responses can vary between runs.
- The comparison does not establish causation or factual accuracy.
