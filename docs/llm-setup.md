# 🤖 AI Summary, Chat, Debate, Comparison, Remix, Relevance & Auto-Tag

Generate paper summaries, chat with papers, debate a paper, compare selected papers, remix papers into new ideas, score relevance, and auto-suggest tags using either an LLM CLI tool or an OpenAI-compatible HTTP endpoint.

## Setup

Add a preset to your `config.json`:

```json
{
  "llm_preset": "copilot",
  "llm_timeout": 120,
  "llm_max_retries": 1
}
```

**Available presets:**

| Preset | Command |
|--------|---------|
| `claude` | `claude -p {prompt}` |
| `codex` | `codex exec {prompt}` |
| `llm` | `llm {prompt}` |
| `copilot` | `copilot --model gpt-5-mini -p {prompt}` |
| `opencode` | `opencode run -m zai-coding-plan/glm-4.7 -- {prompt}` |

Or use a custom command:

```json
{
  "llm_command": "claude -p {prompt}",
  "llm_prompt_template": "Summarize: {title}\n\n{paper_content}",
  "allow_llm_shell_fallback": true,
  "llm_timeout": 180,
  "llm_max_retries": 2
}
```

If both `llm_command` and `llm_preset` are set, `llm_command` wins. A custom `llm_command` must include the `{prompt}` placeholder (where the assembled prompt is substituted); `arxiv-viewer doctor` flags commands that omit it.

**Prompt placeholders:** `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`.

The `{paper_content}` placeholder is replaced with full-paper text when available. The app tries arXiv HTML first, then PDF text extraction, then falls back to the abstract. Extracted full-paper text is cached in `cache.db` for `paper_content_cache_ttl_days` days (default `7`). Set `"paper_content_pdf_fallback": false` to disable PDF text fallback, or `"allow_llm_shell_fallback": false` to block commands that require shell parsing.

## HTTP / OpenAI-Compatible Provider

Use this path for hosted APIs or local servers that expose `/v1/chat/completions`.

```json
{
  "llm_provider_type": "http",
  "llm_api_base_url": "https://api.openai.com",
  "llm_api_key": "YOUR_API_KEY",
  "llm_api_model": "gpt-4o-mini",
  "llm_timeout": 120,
  "llm_max_retries": 1
}
```

Local examples:

```json
{
  "llm_provider_type": "http",
  "llm_api_base_url": "http://localhost:11434",
  "llm_api_model": "llama3.1"
}
```

```json
{
  "llm_provider_type": "http",
  "llm_api_base_url": "http://localhost:1234",
  "llm_api_model": "local-model"
}
```

- OpenAI and compatible hosted APIs usually require `llm_api_key`.
- Ollama, LM Studio, vLLM, and other local servers often leave `llm_api_key` empty.
- `llm_api_base_url` should not include `/v1/chat/completions`; the app appends that path.
- `arxiv-viewer doctor` checks that the base URL and model are present for HTTP mode.

## Streaming Output

Streaming is opt-in so existing CLI and HTTP behavior stays single-shot by default:

```json
{
  "llm_streaming_enabled": true
}
```

When enabled, summaries and chat update incrementally if the provider supports streaming. The HTTP provider uses OpenAI-compatible chat-completions Server-Sent Events with `stream: true`; the CLI provider streams stdout chunks as the subprocess writes them. Summary cache writes still happen only after a stream completes successfully.

## Timeout & Retry Tuning

LLM calls default to a `120` second timeout and `1` retry for transient failures. Tune them in `config.json`:

```json
{
  "llm_timeout": 180,
  "llm_max_retries": 2
}
```

- `llm_timeout`: seconds per attempt (`10..600`)
- `llm_max_retries`: retry budget for timeouts or non-zero exits (`0..5`)

If an LLM action fails unexpectedly, run `arxiv-viewer doctor` to confirm the resolved CLI preset/command or HTTP provider settings.

## Trust Flow

Custom `llm_command` values prompt for trust on first execution. The accepted command hash is stored in config so you aren't prompted again.

## AI Summaries (`Ctrl+s`)

Press `Ctrl+s` on any paper to generate a summary. A mode selector lets you choose:

| Key | Mode | Description |
|-----|------|-------------|
| `d` | Default | Full explanatory summary |
| `q` | Quick | Abstract-focused summary |
| `t` | TLDR | 1-2 sentence summary |
| `m` | Methods | Methods deep-dive |
| `r` | Results | Results-focused summary |
| `c` | Comparison | Related-work comparison |
| `5` | ELI5 | Jargon-free analogy explanation |
| `p` | PhD | Explain for a PhD in another field |

Summaries are cached in the local `cache.db` and persist across sessions.
The PhD mode uses `"physics"` as its default comparison field. Set
`llm_phd_explainer_field` in `config.json` to change it, for example
`"quantum physics"` or `"economics"`.

## Paper Chat (`C`)

Press `C` to start an interactive conversation with the current paper. The full paper content is provided as context to the LLM.

## Debate Paper (`Ctrl+p` -> Debate Paper)

Open the command palette with `Ctrl+p` and run **Debate Paper** on the current paper to generate a threaded two-role review: an enthusiastic advocate makes the strongest fair case for the paper, then Reviewer 2 critiques the claims, baselines, evaluation, assumptions, reproducibility, and missing evidence. Debate uses full paper context when available, falls back to the abstract, and does not cache or save the result.

## Paper Comparison (`Ctrl+v`)

Select exactly 2-3 papers with `Space`, then press `Ctrl+v` to open a side-by-side comparison. The local view appears immediately and works without LLM configuration: it aligns each paper's title, authors, date, categories, comments, and abstract.

Inside the comparison modal, press `g` to generate an optional AI comparison. The LLM comparison uses full paper context when available and returns structured sections for methods, results, key differences, and the bottom line. If no `llm_preset` or `llm_command` is configured, the modal stays open and the app shows the usual LLM setup warning.

## Paper Remix (`Ctrl+p` -> Paper Remix)

Select exactly 2-3 papers, open the command palette with `Ctrl+p`, and run **Paper Remix** to generate one concise research direction that combines their approaches. Remix uses titles, authors, categories, abstracts, and your configured `research_interests`; it does not fetch full paper text or save the result.

## Relevance Scoring (`L`)

Score all loaded papers 1-10 based on your research interests.

```json
{
  "research_interests": "efficient LLM inference, quantization, speculative decoding"
}
```

- Press `L` to score all papers
- Press `Ctrl+l` to edit interests (clears cached scores)
- Cycle the sort order with `s` until it reaches the Relevance mode
- Smart Reading Queue sort also uses these relevance scores as its strongest signal
- Score badges: 🟢 green (8-10), 🟡 yellow (5-7), dim (1-4)

## Auto-Tag (`Ctrl+g`)

Press `Ctrl+g` to have the LLM suggest tags for the current or selected papers based on their content.

## Quick Reference

| Key | Action |
|-----|--------|
| `Ctrl+s` | Generate AI summary (mode selector) |
| `C` | Chat with current paper |
| `Ctrl+v` | Compare 2-3 selected papers locally; press `g` inside for AI comparison |
| `Ctrl+p` | Open command palette; run Debate Paper or Paper Remix |
| `L` | Score all papers by relevance |
| `Ctrl+l` | Edit research interests |
| `Ctrl+g` | Auto-tag current/selected papers |

---

**See also:** [Config Reference](config-reference.md) · [Search & Filters](search-filters.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
