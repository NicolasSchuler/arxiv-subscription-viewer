# ⚙️ Configuration Reference

Complete schema reference for `config.json`. The app reads and writes this file automatically — you can also edit it by hand while the app is not running.

## File Location

Determined by [platformdirs](https://pypi.org/project/platformdirs/):

| Platform | Path |
|----------|------|
| Linux | `~/.config/arxiv-browser/config.json` |
| macOS | `~/Library/Application Support/arxiv-browser/config.json` |
| Windows | `%APPDATA%/arxiv-browser/config.json` |

The file is created on first run. If it becomes corrupt (invalid JSON), the app backs it up to `config.json.corrupt` and starts with defaults.

## General

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | `int` | `1` | Config schema version. Managed by the app. |
| `config_defaulted` | `bool` | `false` | Set to `true` when a corrupt config was replaced with defaults. Not persisted. |
| `onboarding_seen` | `bool` | `false` | Tracks whether the first-run help overlay has been dismissed. Managed by the app. |
| `show_abstract_preview` | `bool` | `false` | Show inline abstract preview in the paper list. Toggle with `p`. |
| `detail_mode` | `string` | `"scan"` | Detail pane density. `"scan"` (compact) or `"full"` (expanded). |
| `prefer_pdf_url` | `bool` | `false` | When `true`, `o` opens the PDF URL instead of the abstract page. |
| `arxiv_api_max_results` | `int` | `50` | Default page size for arXiv API searches (`A`). Clamped to `1..200`. |
| `research_interests` | `string` | `""` | Free-text description of your research interests for LLM relevance scoring (`L`). |
| `category_colors` | `dict[str, str]` | `{}` | Custom category → color overrides (e.g. `{"cs.AI": "green"}`). |

## LLM

Configure AI summaries, paper chat, relevance scoring, and auto-tagging. See [llm-setup.md](llm-setup.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_preset` | `string` | `""` | Named preset: `"claude"`, `"codex"`, `"llm"`, `"copilot"`, `"opencode"`, or `""`. Used when `llm_command` is empty. |
| `llm_command` | `string` | `""` | Shell command template, e.g. `"claude -p {prompt}"`. Takes precedence over `llm_preset` when non-empty. |
| `llm_prompt_template` | `string` | `""` | Custom prompt template. Placeholders: `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`. Empty uses the built-in default. |
| `allow_llm_shell_fallback` | `bool` | `true` | Allow commands that require shell parsing. Set `false` to reject shell-only templates. |
| `llm_max_retries` | `int` | `1` | Retries for transient LLM failures such as timeouts or non-zero exits. Range `0..5` (`1` means up to 2 total attempts). |
| `llm_timeout` | `int` | `120` | Seconds to wait for each LLM attempt before timing out. Range `10..600`. |
| `llm_provider_type` | `string` | `"cli"` | LLM provider backend: `"cli"` (shell command) or `"http"` (OpenAI-compatible API). |
| `llm_api_base_url` | `string` | `""` | Base URL for the HTTP provider, e.g. `"https://api.openai.com"` or `"http://localhost:11434"`. Required when `llm_provider_type` is `"http"`. |
| `llm_api_key` | `string` | `""` | API key for the HTTP provider. Leave empty for local models that don't require auth. |
| `llm_api_model` | `string` | `""` | Model name for the HTTP provider, e.g. `"gpt-4o"`, `"llama3"`. |
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

## Export & PDF

Configure file exports and PDF handling. See [export.md](export.md) for detailed usage.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bibtex_export_dir` | `string` | `""` | Directory for file exports (BibTeX, Markdown, RIS, CSV). Empty defaults to `~/arxiv-exports/`. |
| `pdf_download_dir` | `string` | `""` | Directory for downloaded PDFs (`d`). Empty defaults to `~/arxiv-pdfs/`. |
| `pdf_viewer` | `string` | `""` | External PDF viewer command, e.g. `"zathura {url}"` or `"open -a Skim {path}"`. Empty opens in browser. |
| `trusted_pdf_viewer_hashes` | `list[str]` | `[]` | SHA-256 hashes of PDF viewer commands the user has approved. **Managed by the app** — do not edit. |

## UI & Theme

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `theme_name` | `string` | `"monokai"` | Active color theme. Serialized values: `"monokai"`, `"catppuccin-mocha"`, `"solarized-dark"`, `"high-contrast"`. UI labels: Monokai, Catppuccin, Solarized, High Contrast. Cycle with `Ctrl+t`. |
| `theme` | `dict[str, str]` | `{}` | Legacy theme color overrides. Prefer `theme_name` for new configs. |
| `collapsed_sections` | `list[str]` | `["tags", "relevance", "summary", "s2", "hf", "version"]` | Detail pane sections that start collapsed. Valid keys: `"authors"`, `"abstract"`, `"tags"`, `"relevance"`, `"summary"`, `"s2"`, `"hf"`, `"version"`. Toggle with `Ctrl+d`. |

## Session State

Restored automatically on next run. **Managed by the app** — generally no need to edit by hand.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `session.scroll_index` | `int` | `0` | Last scroll position in the paper list. |
| `session.current_filter` | `string` | `""` | Last active search query. |
| `session.sort_index` | `int` | `0` | Index into sort order cycle: `0`=title, `1`=date, `2`=arxiv_id, `3`=citations, `4`=trending, `5`=relevance. |
| `session.selected_ids` | `list[str]` | `[]` | arXiv IDs of papers selected at exit. |
| `session.current_date` | `string \| null` | `null` | Current date in history mode (`YYYY-MM-DD`), or `null` for non-history mode. |

## User Data

Collections of user-created data. **Managed by the app** via dedicated UI — editing by hand is possible but fragile.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paper_metadata` | `dict[str, object]` | `{}` | Per-paper annotations keyed by arXiv ID. Each entry has: `notes` (string), `tags` (list of strings), `is_read` (bool), `starred` (bool), `last_checked_version` (int or null). |
| `watch_list` | `list[object]` | `[]` | Patterns to highlight. Each entry: `pattern` (string), `match_type` (`"author"` \| `"keyword"` \| `"title"`), `case_sensitive` (bool). Manage with `W`. |
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

## See Also

- [llm-setup.md](llm-setup.md) — AI summary, chat, relevance & auto-tag setup
- [semantic-scholar.md](semantic-scholar.md) — Citation data, recommendations & citation graph
- [huggingface.md](huggingface.md) — HuggingFace trending integration
- [export.md](export.md) — Export formats, PDF download & viewer configuration
- [search-filters.md](search-filters.md) — Query syntax and search bookmarks
- [history-mode.md](history-mode.md) — History directory setup and date navigation
