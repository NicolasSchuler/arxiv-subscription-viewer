# ⚙️ Configuration Reference

Complete schema reference for `config.json`. The app reads and writes this file automatically — you can also edit it by hand while the app is not running.

Use this page when you are ready to tune the workflow after the first scan:

1. Install the CLI and run `arxiv-viewer doctor`.
2. Scan papers with `arxiv-viewer search ...` or `arxiv-viewer browse`.
3. Enrich papers with LLM, Semantic Scholar, HuggingFace, semantic search, or conference-deadline features.
4. Organize local review state with stars, tags, notes, bookmarks, collections, marks, and spaced review.
5. Export papers, then configure paths, API keys, and themes once you know your defaults.

## File Location

Determined by [platformdirs](https://pypi.org/project/platformdirs/):

| Platform | Path |
|----------|------|
| Linux | `~/.config/arxiv-browser/config.json` |
| macOS | `~/Library/Application Support/arxiv-browser/config.json` |
| Windows | `%APPDATA%/arxiv-browser/config.json` |

Run `arxiv-viewer config-path` to print the exact path for the current machine.

## Loading And Recovery Behavior

- The config file is created on the first successful save. A missing file means "use defaults".
- Reads are tolerant: missing keys use defaults, wrong-type values are ignored or sanitized, and malformed collection/watch entries are skipped where possible.
- Invalid JSON or an invalid top-level structure is backed up to `config.json.corrupt`; the app then starts from defaults and marks the in-memory config as defaulted.
- Saves are atomic: the app writes a temporary file in the same directory, then replaces `config.json`.
- Avoid editing `config.json` while the app is running because session state is written on exit and can overwrite manual changes.

## Sanitization And Bounds

The loader validates data before it reaches the TUI. Important rules:

| Area | Rule |
|------|------|
| Type checks | Strings, booleans, integers, lists, and dicts must match the expected JSON type; invalid scalar values fall back to defaults. |
| Integers | JSON booleans are not accepted as integers. |
| `arxiv_api_max_results` | Clamped to `1..200`. |
| `llm_max_retries` | Clamped to `0..5`. |
| `llm_timeout` | Clamped to `10..600` seconds. |
| `paper_content_cache_ttl_days` | Clamped to `1..365`. |
| `pdf_preview_max_pages` | Clamped to `1..20`. |
| `semantic_search_top_k` | Clamped to `1..500`. |
| `semantic_search_min_score` | Clamped to `0..100`. |
| `conference_deadlines_cache_ttl_hours` | Clamped to `1..168`. |
| Session sort index | Reset to `0` when outside the known sort range. |
| `detail_mode` | Must be `"scan"` or `"full"`; otherwise resets to `"scan"`. |
| `collapsed_sections` | Keeps only known detail-section keys. |
| `watch_list[].match_type` | Must be `"author"`, `"title"`, or `"keyword"`; invalid values default to `"author"`. |
| `collections` | Limited to 20 collections and 500 paper IDs per collection. |
| `llm_prompt_template` | Unknown placeholders are rejected and the built-in prompt is used. |

## API Keys And Security Notes

- `llm_api_key`, `s2_api_key`, and `semantic_search_api_key` are stored in plain JSON on your local machine. Protect the config directory with normal OS permissions and do not commit or share the file.
- `llm_command` and `pdf_viewer` can execute local commands. The app records trusted command hashes in `trusted_llm_command_hashes` and `trusted_pdf_viewer_hashes`; those lists are managed by the app and should not be edited by hand.
- Prefer `llm_preset` when one of the built-in CLI providers is enough. Use `llm_command` only when you need a custom local command template.
- For local HTTP services, leave API key fields empty when the endpoint does not require authentication.

## General

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | `int` | `1` | Config schema version. Managed by the app. |
| `config_defaulted` | `bool` | `false` | Set to `true` when a corrupt config was replaced with defaults. Not persisted. |
| `onboarding_seen` | `bool` | `false` | Tracks whether the first-run help overlay has been dismissed. Managed by the app. |
| `last_seen_whats_new` | `string` | `""` | Tag of the last "What's New" release notes the user dismissed. Managed by the app. |
| `show_abstract_preview` | `bool` | `false` | Show inline abstract preview in the paper list. Toggle with `p`. |
| `compact_list` | `bool` | `false` | Show one line (title only) per paper in the list for faster title skimming. Toggle with `z`. |
| `detail_mode` | `string` | `"scan"` | Detail pane density. `"scan"` (compact) or `"full"` (expanded). |
| `prefer_pdf_url` | `bool` | `false` | When `true`, `o` opens the PDF URL instead of the abstract page. |
| `arxiv_api_max_results` | `int` | `50` | Default page size for arXiv API searches (`A`). Clamped to `1..200`. |
| `research_interests` | `string` | `""` | Free-text description of your research interests for LLM relevance scoring (`L`). |
| `category_colors` | `dict[str, str]` | `{}` | Custom category → color overrides (e.g. `{"cs.AI": "green"}`). |

## LLM

Configure AI summaries, paper chat, comparison, relevance scoring, and auto-tagging. See [llm-setup.md](llm-setup.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_preset` | `string` | `""` | Named preset: `"claude"`, `"codex"`, `"llm"`, `"copilot"`, `"opencode"`, or `""`. Used when `llm_command` is empty. |
| `llm_command` | `string` | `""` | Shell command template, e.g. `"claude -p {prompt}"`. Takes precedence over `llm_preset` when non-empty. |
| `llm_prompt_template` | `string` | `""` | Custom prompt template. Placeholders: `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`. Empty uses the built-in default. |
| `llm_phd_explainer_field` | `string` | `"physics"` | Target outside field for the PhD summary mode, e.g. `"quantum physics"` or `"economics"`. |
| `allow_llm_shell_fallback` | `bool` | `true` | Allow commands that require shell parsing. Set `false` to reject shell-only templates. |
| `llm_max_retries` | `int` | `1` | Retries for transient LLM failures such as timeouts or non-zero exits. Range `0..5` (`1` means up to 2 total attempts). |
| `llm_timeout` | `int` | `120` | Seconds to wait for each LLM attempt before timing out. Range `10..600`. |
| `llm_streaming_enabled` | `bool` | `false` | Opt in to incremental summary/chat output for providers that support streaming. Single-shot output remains the default. |
| `llm_provider_type` | `string` | `"cli"` | LLM provider backend: `"cli"` (shell command) or `"http"` (OpenAI-compatible API). |
| `llm_api_base_url` | `string` | `""` | Base URL for the HTTP provider, e.g. `"https://api.openai.com"` or `"http://localhost:11434"`. Required when `llm_provider_type` is `"http"`. |
| `llm_api_key` | `string` | `""` | API key for the HTTP provider. Leave empty for local models that don't require auth. |
| `llm_api_model` | `string` | `""` | Model name for the HTTP provider, e.g. `"gpt-4o"`, `"llama3"`. |
| `paper_content_cache_ttl_days` | `int` | `7` | Days to cache extracted full-paper text used by LLM summaries and chat. Range `1..365`. |
| `paper_content_pdf_fallback` | `bool` | `true` | When arXiv HTML text is unavailable, download/read the PDF and extract text before falling back to the abstract. |
| `trusted_llm_command_hashes` | `list[str]` | `[]` | SHA-256 hashes of LLM commands the user has approved. **Managed by the app** — do not edit. |

## Semantic Scholar

Enrich papers with citation counts, TLDRs, recommendations, and citation graphs. See [semantic-scholar.md](semantic-scholar.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `s2_enabled` | `bool` | `false` | Enable Semantic Scholar enrichment on startup. Toggle at runtime with `Ctrl+e`. |
| `s2_api_key` | `string` | `""` | Optional API key for higher rate limits. Get one at [semanticscholar.org](https://www.semanticscholar.org/product/api#api-key). |
| `s2_cache_ttl_days` | `int` | `7` | Days to cache S2 data in the local SQLite database. |

## HuggingFace

Surface trending signals from HuggingFace Daily Papers. See [huggingface.md](huggingface.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hf_enabled` | `bool` | `false` | Enable HuggingFace trending on startup. Toggle at runtime with `Ctrl+h`. |
| `hf_cache_ttl_hours` | `int` | `6` | Hours to cache HF daily data (trending changes frequently). |

## Semantic Search

Optional local or sidecar embedding search for `~`-prefixed local queries. Embeddings are cached in `cache.db`; if no backend is available, the app falls back to fuzzy search.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `semantic_search_backend` | `string` | `"auto"` | Embedding backend: `"auto"`, `"fastembed"`, `"sentence-transformers"`, `"http"`, or `"off"`. |
| `semantic_search_model` | `string` | `"BAAI/bge-small-en-v1.5"` | Embedding model id. Use `"Qwen/Qwen3-Embedding-0.6B"` with the `sentence-transformers` or HTTP backend for Qwen. |
| `semantic_search_api_base_url` | `string` | `""` | Base URL for an OpenAI-compatible embeddings endpoint, e.g. a local TEI server. |
| `semantic_search_api_key` | `string` | `""` | API key for the embeddings endpoint. Leave empty for local servers without auth. |
| `semantic_search_top_k` | `int` | `100` | Maximum semantic results to show. Clamped to `1..500`. |
| `semantic_search_min_score` | `int` | `15` | Minimum cosine similarity score as a percentage after negative similarities are clamped to zero. Range `0..100`. |

## Conference Deadlines

Import upcoming submission deadlines from the third-party, community-maintained AI Deadlines dataset. Dates are cached locally and matched to papers by arXiv category, tags, title, and abstract overlap.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `conference_deadlines_enabled` | `bool` | `false` | Load cached/imported conference deadlines on startup and show matching future submission targets in paper details. |
| `conference_deadlines_source_url` | `string` | AI Deadlines YAML URL | Source URL for the AI Deadlines-compatible YAML feed. Override this for a local mirror or custom curated list. |
| `conference_deadlines_cache_ttl_hours` | `int` | `24` | Hours to reuse the imported deadline snapshot before fetching again. Clamped to `1..168`. |

## Export & PDF

Configure file exports and PDF handling. See [export.md](export.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bibtex_export_dir` | `string` | `""` | Directory for file exports (BibTeX, Markdown, RIS, CSV). Empty defaults to `~/arxiv-exports/`. |
| `pdf_download_dir` | `string` | `""` | Directory for downloaded PDFs (`d`). Empty defaults to `~/arxiv-pdfs/`. |
| `pdf_preview_max_pages` | `int` | `3` | Number of pages rendered in the terminal PDF preview (`F`). Range `1..20`. |
| `pdf_viewer` | `string` | `""` | External PDF viewer command, e.g. `"zathura {url}"` or `"open -a Skim {path}"`. Empty opens in browser. |
| `trusted_pdf_viewer_hashes` | `list[str]` | `[]` | SHA-256 hashes of PDF viewer commands the user has approved. **Managed by the app** — do not edit. |

## UI & Theme

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `theme_name` | `string` | `"monokai"` | Active color theme. Built-ins: `"monokai"`, `"catppuccin-mocha"`, `"solarized-dark"`, `"solarized-light"`, `"high-contrast"`, `"dracula"`, `"nord"`, `"gruvbox-dark"`, `"tokyo-night"`, `"everforest-dark"`, `"github-light"`. Can also name an entry in `custom_themes`. Cycle with `Ctrl+t`. |
| `theme` | `dict[str, str]` | `{}` | Per-color overrides layered over the active built-in or custom theme. |
| `custom_themes` | `dict[str, dict[str, str]]` | `{}` | Named user palettes. Each palette may override any key from the built-in theme color map; omitted keys inherit from Monokai. Custom names appear after built-ins when cycling with `Ctrl+t`. |
| `collapsed_sections` | `list[str]` | `["tags", "relevance", "summary", "s2", "hf", "version"]` | Detail pane sections that start collapsed. Valid keys: `"authors"`, `"abstract"`, `"tags"`, `"relevance"`, `"deadlines"`, `"summary"`, `"s2"`, `"hf"`, `"version"`. Toggle with `Ctrl+d`. |

### Custom Theme Example

Set `theme_name` to a custom palette name, then define that name under `custom_themes`.

```json
{
  "theme_name": "paper-night",
  "custom_themes": {
    "paper-night": {
      "background": "#11131a",
      "panel": "#181b24",
      "text": "#f2f0e8",
      "muted": "#b9b4a8",
      "accent": "#7dcfff",
      "green": "#9ece6a",
      "orange": "#ff9e64",
      "pink": "#f7768e",
      "purple": "#bb9af7"
    }
  }
}
```

### `user.tcss`

Advanced users can create a `user.tcss` file next to `config.json` to layer custom Textual CSS over the embedded app stylesheet:

| Platform | Path |
|----------|------|
| Linux | `~/.config/arxiv-browser/user.tcss` |
| macOS | `~/Library/Application Support/arxiv-browser/user.tcss` |
| Windows | `%APPDATA%/arxiv-browser/user.tcss` |

The file is optional and loaded after the built-in CSS, so selectors in `user.tcss` win. Keep overrides small: internal widget IDs and classes may change across releases.

```tcss
/* Brighter accent for the current theme */
Screen {
    --th-accent: #ffd75f;
}

/* Slightly darker details pane */
PaperDetails {
    background: #181818;
}
```

## Session State

Restored automatically on next run. **Managed by the app** — generally no need to edit by hand.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `session.scroll_index` | `int` | `0` | Last scroll position in the paper list. |
| `session.current_filter` | `string` | `""` | Last active search query. |
| `session.sort_index` | `int` | `0` | Index into sort order cycle: `0`=title, `1`=date, `2`=arxiv_id, `3`=citations, `4`=trending, `5`=relevance, `6`=queue, `7`=triage. |
| `session.selected_ids` | `list[str]` | `[]` | arXiv IDs of papers selected at exit. |
| `session.current_date` | `string \| null` | `null` | Current date in history mode (`YYYY-MM-DD`), or `null` for non-history mode. |

## User Data

Collections of user-created data. **Managed by the app** via dedicated UI — editing by hand is possible but fragile.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paper_metadata` | `dict[str, object]` | `{}` | Per-paper annotations keyed by arXiv ID. Each entry has: `notes` (string), `tags` (list of strings), `is_read` (bool), `starred` (bool), `last_checked_version` (int or null), `next_review_date` (`YYYY-MM-DD` string or null), `review_stage` (interval index or null), and `line_annotations` (list of `{line, text}` objects anchored to visible detail-pane lines). Review fields are managed by the app and use the interval sequence 1, 3, 7, 14, 30 days. |
| `watch_list` | `list[object]` | `[]` | Patterns to highlight. Each entry: `pattern` (string), `match_type` (`"author"` \| `"keyword"` \| `"title"`), `case_sensitive` (bool). Manage with `W`. |
| `tracked_authors` | `list[str]` | `[]` | Exact author names to highlight across loaded papers. Manage from `Ctrl+p` → **Track Author**; matching uses normalized full-name equality and appears through the watched-paper highlight path. |
| `bookmarks` | `list[object]` | `[]` | Saved search queries (max 9). Each entry: `name` (string), `query` (string). Manage with `Ctrl+b` / `Ctrl+Shift+b`. |
| `collections` | `list[object]` | `[]` | Paper reading lists (max 20, 500 papers each). Each entry: `name` (string), `description` (string), `paper_ids` (list of strings), `created` (ISO 8601 timestamp). Manage with `Ctrl+k`. |
| `marks` | `dict[str, str]` | `{}` | Named marks mapping a letter (`a`-`z`) to an arXiv ID. Set with `m`, jump with `'`. |

## Minimal Example

```json
{
  "llm_preset": "copilot",
  "s2_enabled": true,
  "hf_enabled": true,
  "research_interests": "efficient LLM inference, quantization, speculative decoding",
  "bibtex_export_dir": "~/papers/exports",
  "pdf_download_dir": "~/papers/pdfs"
}
```

All other keys use their defaults and are populated by the app as needed.

## Examples

### Live Search With Enrichment

```json
{
  "arxiv_api_max_results": 100,
  "s2_enabled": true,
  "hf_enabled": true,
  "research_interests": "efficient LLM inference, quantization, speculative decoding"
}
```

### Local HTTP LLM Provider

```json
{
  "llm_provider_type": "http",
  "llm_api_base_url": "http://localhost:11434",
  "llm_api_model": "llama3",
  "llm_timeout": 180,
  "llm_streaming_enabled": true
}
```

### Export And PDF Defaults

```json
{
  "bibtex_export_dir": "~/papers/exports",
  "pdf_download_dir": "~/papers/pdfs",
  "pdf_preview_max_pages": 5,
  "prefer_pdf_url": true
}
```

## See Also

- [llm-setup.md](llm-setup.md) — AI summary, chat, comparison, relevance & auto-tag setup
- [semantic-scholar.md](semantic-scholar.md) — Citation data, recommendations & citation graph
- [huggingface.md](huggingface.md) — HuggingFace trending integration
- [export.md](export.md) — Export formats, PDF download & viewer configuration
- [search-filters.md](search-filters.md) — Query syntax and search bookmarks
- [history-mode.md](history-mode.md) — History directory setup and date navigation
