# Automated Markdown Digests

Use `arxiv-viewer digest` when you want a non-interactive daily or weekly paper brief that can run from cron. The command prints Markdown to stdout by default, or writes it to a file with `--output`.

Add `--tui` when you want the same digest source and enrichment sections as an
interactive inbox instead of Markdown output.

## Quick Examples

```bash
# Daily digest for the newest cs.AI arXiv day
arxiv-viewer digest --category cs.AI

# Weekly digest for a topic
arxiv-viewer digest --query "diffusion transformer" --field title --period weekly

# Render a saved local digest file instead of fetching live results
arxiv-viewer digest --input history/2026-02-13.txt

# Save Markdown to a file
arxiv-viewer digest --category cs.CL --period weekly --output ~/digests/arxiv-weekly.md

# Open a weekly digest as an interactive inbox
arxiv-viewer digest --category cs.AI --period weekly --include-triage --tui
```

## Cron

The digest command does not require an interactive terminal. Stdout is reserved for Markdown; warnings and non-fatal enrichment failures go to stderr.

```cron
# Every weekday morning, overwrite the latest daily digest
30 7 * * 1-5 cd "$HOME/research/arxiv" && arxiv-viewer digest --category cs.AI --output "$HOME/digests/arxiv-daily.md"
```

For email or Slack in v1, pipe the generated Markdown to your own delivery tool:

```bash
arxiv-viewer digest --category cs.AI \
  | mail -s "Daily arXiv digest" you@example.com

arxiv-viewer digest --category cs.AI \
  | jq -Rs '{text: .}' \
  | curl -X POST -H 'Content-Type: application/json' --data @- "$SLACK_WEBHOOK_URL"
```

## TUI Inbox

`--tui` runs the same source loading and optional enrichment pipeline, then opens
the existing browser with papers ordered by digest section priority. Rows show
ephemeral `Inbox:` badges such as `Inbox:High Relevance`, `Inbox:Likely Star`,
or `Inbox:Trending on Hugging Face`.

```bash
arxiv-viewer digest --category cs.AI --period weekly --tui
arxiv-viewer digest --input history/2026-02-13.txt --include-triage --tui
```

The inbox is intentionally flat, so normal search, sorting, quick triage,
enrichment, selection, and export workflows still work. Inbox badges are not
saved as tags and the inbox launch does not restore or overwrite the regular
browse session. `--tui` requires an interactive terminal and cannot be combined
with `--output`.

## Sources

Live mode uses the arXiv API:

```bash
arxiv-viewer digest --query "agent benchmark" --category cs.LG --period weekly
```

`--period daily` collects the newest matching submitted day. `--period weekly` collects the seven calendar days ending at the newest matching submitted day.

File mode renders exactly the parsed file:

```bash
arxiv-viewer digest --input history/2026-02-13.txt
```

Do not combine `--input` with live source flags such as `--query`, `--category`, `--field`, `--period`, or `--max-results`.

## Sections

Digests include these sections when data is available:

- Overview: total papers, top categories, watch-list match count, read/starred count.
- Watch List Matches: papers matching configured author, title, or keyword watch entries.
- High Relevance: papers at or above `--min-relevance` from cached or freshly generated LLM scores.
- Likely Star / Unsure Review Queue: optional sections from the local sklearn triage model.
- Trending on Hugging Face: loaded papers that appear in Hugging Face Daily Papers.
- Version Updates: starred papers with newer arXiv versions.
- New Papers: the loaded source papers.

Use `--limit N` to cap each list section. Omitted counts are shown when more papers are available.

## Relevance

By default, digest generation loads cached relevance scores and scores missing papers only when both of these are configured:

- `research_interests`
- an LLM provider, either CLI or HTTP

Fresh scoring for a custom `llm_command` runs only after the command has already been trusted in the TUI. If it is not trusted, the digest uses cached scores and prints a warning to stderr. Presets and HTTP providers do not need the custom-command trust prompt.

Useful controls:

```bash
arxiv-viewer digest --category cs.AI --cached-relevance-only
arxiv-viewer digest --category cs.AI --no-relevance
arxiv-viewer digest --category cs.AI --min-relevance 8
```

## Local Triage Model

If you trained the local model from the TUI command palette, include its buckets in headless digests:

```bash
arxiv-viewer digest --category cs.AI --include-triage
```

This adds **Likely Star** and **Unsure Review Queue** sections. It keeps **New Papers** intact and never hides likely-skip papers by default. If sklearn support or the trained model artifact is missing, the digest continues and prints a warning to stderr.

## Hugging Face

By default, Hugging Face trending follows `hf_enabled` from `config.json`.

```bash
arxiv-viewer digest --category cs.AI --include-hf
arxiv-viewer digest --category cs.AI --no-hf
```

HF data uses the existing daily cache and `hf_cache_ttl_hours`. If the network fetch fails, digest generation continues and prints a warning to stderr.

## Version Updates

Version checks follow the existing app behavior: only starred papers are checked. The command updates `last_checked_version` baselines in `config.json` when the check succeeds, and renders updates such as `v1 -> v3`.

Disable version checks for fully offline or faster cron runs:

```bash
arxiv-viewer digest --category cs.AI --no-versions
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Digest generated, even if optional enrichment warned or no papers matched. |
| `1` | Source, parse, network, or output-write failure prevented digest generation. |
| `2` | Usage error, such as conflicting `--input` and live source flags. |
