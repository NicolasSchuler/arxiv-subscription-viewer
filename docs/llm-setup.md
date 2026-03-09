# 🤖 AI Summary, Chat, Relevance & Auto-Tag

Generate paper summaries, chat with papers, score relevance, and auto-suggest tags — all powered by any LLM CLI tool you have installed.

## Setup

Add a preset to your `config.json`:

```json
{ "llm_preset": "copilot" }
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
  "allow_llm_shell_fallback": true
}
```

**Prompt placeholders:** `{title}`, `{authors}`, `{categories}`, `{abstract}`, `{arxiv_id}`, `{paper_content}`.

The `{paper_content}` placeholder is replaced with the full paper text (fetched from arXiv HTML), falling back to the abstract if unavailable. Set `"allow_llm_shell_fallback": false` to block commands that require shell parsing.

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

Summaries are cached in a local SQLite database and persist across sessions.

## Paper Chat (`C`)

Press `C` to start an interactive conversation with the current paper. The full paper content is provided as context to the LLM.

## Relevance Scoring (`L`)

Score all loaded papers 1-10 based on your research interests.

```json
{
  "research_interests": "efficient LLM inference, quantization, speculative decoding"
}
```

- Press `L` to score all papers
- Press `Ctrl+l` to edit interests (clears cached scores)
- Sort by relevance with `s`
- Score badges: 🟢 green (8-10), 🟡 yellow (5-7), dim (1-4)

## Auto-Tag (`Ctrl+g`)

Press `Ctrl+g` to have the LLM suggest tags for the current or selected papers based on their content.

## Quick Reference

| Key | Action |
|-----|--------|
| `Ctrl+s` | Generate AI summary (mode selector) |
| `C` | Chat with current paper |
| `L` | Score all papers by relevance |
| `Ctrl+l` | Edit research interests |
| `Ctrl+g` | Auto-tag current/selected papers |
