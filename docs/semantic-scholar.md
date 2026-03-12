# 📊 Semantic Scholar Integration

Enrich papers with citation counts, fields of study, TLDRs, S2-powered recommendations, and citation graph exploration.

## Setup

Enable at runtime with `Ctrl+e`, or add to `config.json`:

```json
{
  "s2_enabled": true,
  "s2_api_key": "",
  "s2_cache_ttl_days": 7
}
```

| Key | Description | Default |
|-----|-------------|---------|
| `s2_enabled` | Enable S2 enrichment on startup | `false` |
| `s2_api_key` | Optional API key for higher rate limits | `""` |
| `s2_cache_ttl_days` | Days to cache S2 paper metadata in SQLite | `7` |

## Usage

| Key | Action |
|-----|--------|
| `Ctrl+e` | Toggle S2 on/off |
| `e` | Fetch S2 data for current paper (citation count, TLDR, fields of study) |
| `R` | Recommendations (local or S2-powered when enabled) |
| `G` | Citation graph — drill-down through references and citations |
| `s` | Sort by citations (cycle sort order) |

## Citation Graph

Press `G` on any paper to explore its citation graph. The modal shows a two-panel layout with references and citations, supporting stack-based drill-down navigation with a breadcrumb trail.

All S2 data is cached in SQLite. Paper metadata uses a configurable TTL (`s2_cache_ttl_days`, default 7 days). Recommendations and citation graph data use a fixed 3-day TTL.

---

**See also:** [HuggingFace](huggingface.md) · [Config Reference](config-reference.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
