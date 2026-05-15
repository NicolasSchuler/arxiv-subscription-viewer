# 🛠️ Troubleshooting

Common issues and how to fix them.

> **Quick diagnostic:** Run `arxiv-viewer doctor` to check your environment,
> current working directory, LLM setup, export paths, and more in one command.

This guide assumes both supported entry paths:

- `history/` mode for local dated digest files
- `search` mode for live arXiv API results

## `arxiv-viewer doctor`

Run this from the same directory you normally launch the app from:

```bash
arxiv-viewer doctor
```

It reports:

- config file path/status
- whether `./history/` contains `YYYY-MM-DD.txt` files
- resolved LLM command or preset, including `{prompt}` placeholder checks, shell-fallback policy, and whether the binary is on `PATH`
- whether Semantic Scholar / HuggingFace are enabled
- export/PDF directories and whether they'll be created on first use
- whether the current terminal is an interactive TTY

Exit code is `0` when no warnings are found, `1` when `doctor` reports issues.

## No Papers Loaded

**Symptom:** The viewer opens but the paper list is empty, or `browse` mode loads nothing.

**File not found:**

```
Error: papers.txt not found
```

Check the path you passed with `-i`. In history mode (no `-i`), the viewer looks for `YYYY-MM-DD.txt` files inside a `history/` subdirectory relative to the current working directory. `arxiv-viewer doctor` will report which directory it is inspecting and whether it found any history files.

```bash
# Explicit file
uv run arxiv-viewer -i ~/mail/arxiv-2025-01-15.txt

# History mode — needs history/*.txt files
ls history/
# 2025-01-14.txt  2025-01-15.txt  ...
```

**Empty or wrong format:**

The parser expects arXiv subscription email text pasted into a `.txt` file. Entries are separated by lines of 70+ dashes (`------...------`). Each entry needs at minimum an `arXiv:` ID line. Malformed entries are silently skipped — if none parse, the list will be empty.

Expected entry structure:

```
arXiv:2301.12345
Date: Mon, 15 Jan 2024 (v1)
Title: Example Paper Title
Authors: Alice, Bob
Categories: cs.AI cs.LG
\\
  Abstract text here...
\\
```

**Permission denied:**

```
Error: papers.txt permission denied
```

Check file permissions with `ls -la <file>`.

**Expected search-mode behavior:**

If you do not use local files, prefer:

```bash
arxiv-viewer search --category cs.AI
```

An empty search page is a different problem from missing local history files.

---

## Digest CLI Issues

**Symptom:** `arxiv-viewer digest` prints a digest but some sections are missing.

That usually means the optional data source was unavailable:

- Watch-list sections require configured `watch_list` entries.
- High-relevance sections require cached scores, or `research_interests` plus a configured LLM provider.
- Hugging Face sections require matching Daily Papers data. Use `--include-hf` to fetch even when `hf_enabled` is false.
- Version sections check starred papers only. Use `x` in the TUI to star papers before expecting version-update tracking.

**Untrusted custom LLM command:**

Fresh digest relevance scoring cannot show the TUI trust prompt. If `llm_command` is custom and has not already been approved, the digest uses cached scores only and warns on stderr. Open the TUI and run a relevance action once, or use `--cached-relevance-only` for predictable cron runs.

**No matching papers:**

An empty but successful digest exits `0` and contains `No matching papers.`. For live mode, check `--query`, `--field`, and `--category`. For file mode, confirm the input parses with:

```bash
arxiv-viewer digest --input history/YYYY-MM-DD.txt
```

**Warnings in cron mail:**

Digest Markdown goes to stdout and warnings go to stderr. Redirect them independently:

```bash
arxiv-viewer digest --category cs.AI --output digest.md 2>>digest-warnings.log
```

**Output write failure:**

If `--output` points at a directory, unwritable path, or invalid parent location, the command exits `1`. Check the parent directory and permissions.

---

## LLM Not Working

**Symptom:** `Ctrl+s` (summary), `Ctrl+v` then `g` (AI comparison), `L` (relevance), `Ctrl+g` (auto-tag), or `C` (chat) does nothing or shows an error.

**No preset configured:**

LLM features require either `llm_preset` or `llm_command` in your config. Without one, LLM actions are silently skipped.

```json
{ "llm_preset": "copilot" }
```

Available presets: `claude`, `codex`, `llm`, `copilot`, `opencode`. See [LLM Setup](llm-setup.md) for details.

**Command not found:**

The preset maps to a CLI tool that must be installed and on your `$PATH`. For example, `"llm_preset": "claude"` runs `claude -p {prompt}`. If `claude` isn't installed, the subprocess will fail.

Verify the tool works standalone:

```bash
echo "hello" | claude -p "say hi"
```

**Trust not accepted:**

Custom `llm_command` values prompt for trust approval via a confirmation dialog. If you dismiss the dialog, the action is cancelled. The trust hash is stored in `config.json` under `trusted_llm_command_hashes` so you're only prompted once per unique custom command.

If you change `llm_command`, you'll be prompted again for the new command.

**Timeouts or retries need tuning:**

Each LLM attempt uses `llm_timeout` (default `120`, range `10..600`). Transient failures can retry up to `llm_max_retries` times (default `1`, range `0..5`). Long prompts with `{paper_content}` may need a higher timeout or fewer full-paper requests on slower models.

```json
{
  "llm_timeout": 180,
  "llm_max_retries": 2
}
```

If you're not sure which command the app is actually using, run `arxiv-viewer doctor`.

**Streaming behavior:**

LLM streaming is disabled by default. Enable it only if you want summaries and chat to update incrementally:

```json
{ "llm_streaming_enabled": true }
```

If streamed output looks partial or your provider does not support OpenAI-compatible streaming, set it back to `false`.

**Full-paper content looks abstract-only:**

The app tries arXiv HTML first, then PDF text extraction, then the abstract. If many papers still use abstract-only context, check that PDF downloads are reachable and that PDF fallback is enabled:

```json
{
  "paper_content_pdf_fallback": true,
  "paper_content_cache_ttl_days": 7
}
```

Set `paper_content_cache_ttl_days` lower if you need fresh extraction sooner, or delete `cache.db` only if you intentionally want to clear all cached summaries/enrichment/content.

**Invalid prompt template:**

```
Invalid prompt template: ... Valid placeholders: {title}, {authors}, {categories}, {abstract}, {arxiv_id}, {paper_content}
```

Check your `llm_prompt_template` for typos in placeholder names.

---

## Semantic Scholar Not Showing Data

**Symptom:** No citation counts, TLDRs, or recommendations appear.

**Not enabled:**

Semantic Scholar enrichment is opt-in. Enable it in `config.json`:

```json
{ "s2_enabled": true }
```

Or toggle it at runtime with `Ctrl+e`.

**Paper not found in S2:**

Not all arXiv papers are indexed by Semantic Scholar. When a paper isn't found, the viewer logs an info message and returns no data — this is expected. Check [Semantic Scholar](https://www.semanticscholar.org/) directly to verify.

**API rate limiting:**

Without an API key, you're subject to S2's public rate limits. Requests retry up to 2 times after the initial attempt (3 total attempts) with exponential backoff (1s → 2s). If you hit limits frequently, add an API key:

```json
{
  "s2_enabled": true,
  "s2_api_key": "your-key-here"
}
```

Request a key at [Semantic Scholar API](https://www.semanticscholar.org/product/api#api-key).

**Stale cache:**

S2 data is cached for 7 days (configurable via `s2_cache_ttl_days`). To force a refresh, delete the cache file:

```bash
# macOS
rm ~/Library/Application\ Support/arxiv-browser/semantic_scholar.db

# Linux
rm ~/.config/arxiv-browser/semantic_scholar.db
```

---

## HuggingFace Data Missing

**Symptom:** No trending/upvote indicators on papers.

**Not enabled:**

HuggingFace trending is opt-in:

```json
{ "hf_enabled": true }
```

Or toggle at runtime with `Ctrl+h`.

**Paper not in Daily Papers:**

HuggingFace Daily Papers only tracks a curated subset of arXiv papers. Most papers won't have HF data — this is normal.

**API unreachable:**

The HF API has a 15-second timeout and the same 3-attempt retry budget (1s → 2s backoff). If the endpoint returns 404 or non-JSON data, the viewer falls back to an empty result. Check your network connection if no HF data loads at all.

**Stale cache:**

HF data is cached for 6 hours (configurable via `hf_cache_ttl_hours`). Delete the cache to force a refresh:

```bash
# macOS
rm ~/Library/Application\ Support/arxiv-browser/huggingface.db

# Linux
rm ~/.config/arxiv-browser/huggingface.db
```

---

## Smart Reading Queue Looks Like Date Sort

**Symptom:** `sort:queue` is active, but papers appear mostly ordered by recency or watch-list matches.

Queue mode uses the signals that are currently available. Until you run relevance scoring (`L`), enable HuggingFace (`Ctrl+h`), or enrich papers with Semantic Scholar (`Ctrl+e` / `e`), the ranking is mostly recency plus watch-list matches. Missing enrichment is expected to contribute zero rather than blocking the queue.

---

## Semantic Search Falls Back to Fuzzy

**Symptom:** A `~` query returns keyword-like fuzzy results or shows a semantic-search warning.

Semantic search is optional. Install the lightweight backend:

```bash
pip install "arxiv-subscription-viewer[semantic-fastembed]"
```

For Hugging Face/Qwen models, install `sentence-transformers` separately and configure:

```json
{
  "semantic_search_backend": "sentence-transformers",
  "semantic_search_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

For a local embeddings sidecar, set `semantic_search_backend` to `"http"` and configure `semantic_search_api_base_url`.

Embeddings are cached in `cache.db` in the `semantic_embeddings` table. Delete `cache.db` only if you intentionally want to clear all local caches.

---

## Export Failures

**Symptom:** BibTeX, RIS, CSV, or Markdown export fails or produces no file.

**Directory permissions:**

Exports write to `~/arxiv-exports/` by default (configurable via `bibtex_export_dir`). Ensure the directory exists and is writable:

```bash
mkdir -p ~/arxiv-exports
```

PDF downloads go to `~/arxiv-pdfs/` (configurable via `pdf_download_dir`).

**Path traversal guard:**

The export system validates that output paths stay within the target directory. A crafted arXiv ID containing `../` would be rejected:

```
Invalid arXiv ID for path construction: '../../etc/passwd'
```

This is a security guard — normal arXiv IDs (e.g., `2301.12345`) are never affected.

**Clipboard not available:**

Copy-to-clipboard (`c`) requires a platform clipboard tool:

| Platform | Required tool |
|----------|--------------|
| macOS | `pbcopy` (built-in) |
| Linux | `xclip` or `xsel` |
| Windows | `clip` (built-in) |

On Linux, install `xclip`:

```bash
sudo apt install xclip
```

---

## PDF Viewer Issues

**Symptom:** `P` opens the PDF URL in a browser instead of a local viewer, or the viewer fails to launch.

**Viewer not configured:**

By default, `pdf_viewer` is empty and PDFs open as URLs in your default browser. To use a local viewer, set the command in `config.json`:

```json
{ "pdf_viewer": "zathura {path}" }
```

**Command format:**

The viewer command supports `{url}` and `{path}` placeholders. If neither is present, the URL/path is appended as the last argument.

```json
// Placeholder style
{ "pdf_viewer": "open -a Skim {path}" }

// Append style (equivalent to "evince <url>")
{ "pdf_viewer": "evince" }
```

**Trust prompt:**

Custom viewer commands require trust approval on first use (same flow as LLM commands). If you dismiss the prompt, the action is cancelled.

---

## PDF Preview Issues

**Symptom:** `F` fails to open a preview or shows a render error.

**PDF not downloaded and no network client:**

The preview action uses an existing downloaded PDF when present. If the file is missing and the app cannot download it in the current session, press `d` first or open a view with network access.

**Render cache:**

Preview PNGs are cached in `.preview-cache/` under the configured PDF download directory. If previews look stale or corrupt, delete that subdirectory; it will be recreated on the next preview.

**Large or slow previews:**

Lower the page limit:

```json
{ "pdf_preview_max_pages": 1 }
```

---

## HTML Figure Preview Issues

**Symptom:** Pressing `I` does not show a figure preview.

**arXiv HTML unavailable:**

Not every paper has a rendered HTML page. The app requests `https://arxiv.org/html/{arxiv_id}`; if arXiv returns `404` or another non-OK status, the preview shows a warning and leaves the paper unchanged.

**No figure found:**

The preview intentionally selects the first LaTeXML paper figure (`<figure class="ltx_figure"><img ...>`) so page chrome, logos, and unrelated header images are ignored. Papers with no matching figure show a warning.

**Image fetch or type failure:**

If the figure URL is unreachable, the content type is unsupported, or the bytes are not a readable image, the cache entry is not used and the app reports the failure in the TUI.

**Stale or corrupt cache:**

Figure previews are cached in `.figure-cache/` next to the PDF download/cache location. Corrupt cached images are replaced automatically on the next successful preview. To force a refresh, delete that paper's cached figure PNG from `.figure-cache/`.

**Terminal rendering limitations:**

The preview uses the same terminal-safe renderer as PDF preview. Unicode half-block rendering gives the best result in modern terminals; `--ascii` and no-color modes fall back to lower-fidelity safe characters.

---

## Debug Mode

**Symptom:** Something isn't working and you need more detail.

**Enable debug logging:**

```bash
uv run arxiv-viewer --debug
```

This writes detailed logs to a rotating log file:

| Platform | Log path |
|----------|----------|
| Linux | `~/.config/arxiv-browser/debug.log` |
| macOS | `~/Library/Application Support/arxiv-browser/debug.log` |
| Windows | `%APPDATA%/arxiv-browser/debug.log` |

Logs rotate at 5 MB with 3 backup files (`debug.log.1`, `.2`, `.3`).

**What gets logged:**

- File parsing results and skipped entries
- API requests and responses (S2, HF, arXiv)
- LLM command execution and timeouts
- Config load/save operations
- Cache hits and misses
- Trust hash checks

**Reading logs:**

```bash
# macOS
tail -f ~/Library/Application\ Support/arxiv-browser/debug.log

# Linux
tail -f ~/.config/arxiv-browser/debug.log
```

Without `--debug`, all logging is suppressed (the TUI captures stderr).

---

## Config File Issues

**Symptom:** Settings aren't applied, or the viewer resets to defaults on every launch.

**Config file location:**

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/arxiv-browser/config.json` |
| Linux | `~/.config/arxiv-browser/config.json` |
| Windows | `%APPDATA%/arxiv-browser/config.json` |

**Corrupted config:**

If `config.json` contains invalid JSON or an unexpected structure, the viewer:

1. Logs a warning (visible with `--debug`)
2. Backs up the corrupt file to `config.json.corrupt`
3. Starts with default settings

Check for a backup file:

```bash
# macOS
ls ~/Library/Application\ Support/arxiv-browser/config.json*

# Linux
ls ~/.config/arxiv-browser/config.json*
```

If `config.json.corrupt` exists, your config was malformed. Inspect it to recover any settings:

```bash
cat ~/.config/arxiv-browser/config.json.corrupt | python -m json.tool
```

**Reset to defaults:**

Delete the config file. The viewer will create a fresh one on next launch:

```bash
# macOS
rm ~/Library/Application\ Support/arxiv-browser/config.json

# Linux
rm ~/.config/arxiv-browser/config.json
```

**Reset everything (config + all caches):**

```bash
# macOS
rm -rf ~/Library/Application\ Support/arxiv-browser/

# Linux
rm -rf ~/.config/arxiv-browser/
```

This removes config, LLM summary cache (`summaries.db`), relevance scores (`relevance.db`), S2 cache (`semantic_scholar.db`), HF cache (`huggingface.db`), and semantic embeddings in unified `cache.db`.
