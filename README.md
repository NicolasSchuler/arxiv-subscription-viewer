<p align="center">
  <img src="docs/logo.png" alt="arXiv Subscription Viewer" width="200">
</p>

<h1 align="center">arXiv Subscription Viewer</h1>

<p align="center">
  <b>Browse, search, and manage arXiv papers from your terminal.</b><br>
  Fuzzy search · AI summaries · citation graphs · PDF downloads · export to BibTeX/Markdown/CSV
</p>

<p align="center">
  <img src="docs/screenshot.svg" alt="Screenshot" width="800">
</p>

## 🚀 Getting Started

```bash
# Install
pip install arxiv-subscription-viewer
# or: uv tool install arxiv-subscription-viewer

# Search arXiv API directly
arxiv-viewer search --category cs.AI

# Search by topic
arxiv-viewer search --query "diffusion transformer" --field title

# Paginate through results (instead of collecting the newest day)
arxiv-viewer search --query "attention" --mode page --max-results 100

# Browse from email digest files (see docs/history-mode.md)
arxiv-viewer browse

# List local history dates
arxiv-viewer dates
```

> **Requires Python 3.13+** · Press `?` in-app for help · `Ctrl+p` opens commands
>
> Global options: `--debug` (log to file) · `--ascii` (ASCII-only icons) · `--color auto|always|never` · `--no-color`
>
> Debug log paths: `~/.config/arxiv-browser/debug.log` (Linux) · `~/Library/Application Support/arxiv-browser/debug.log` (macOS) · `%APPDATA%/arxiv-browser/debug.log` (Windows)

## ✨ Highlights

| | Feature | Key |
|---|---------|-----|
| 🔍 | **Fuzzy search** with filters (`cat:cs.AI`, `tag:`, `unread`, `starred`) | `/` |
| 🤖 | **AI summaries, chat & auto-tag** via any LLM CLI (Claude, Copilot, llm, …) | `Ctrl+s` / `C` / `Ctrl+g` |
| 📊 | **Citation graph** and recommendations via Semantic Scholar | `G` / `R` |
| 🔥 | **HuggingFace trending** — upvotes, keywords, GitHub links | `Ctrl+h` |
| 🏷️ | **Tags, notes, stars** — organize your reading | `t` / `n` / `x` |
| 📁 | **Collections** — curate reading lists | `Ctrl+k` |
| 📥 | **Export** — BibTeX, Markdown, RIS, CSV, PDF download | `E` / `d` |
| 🎯 | **Relevance scoring** — LLM scores papers against your interests | `L` |
| 📅 | **History mode** — navigate daily email digests with `[` / `]` | |
| ⌨️ | **Command palette** — quick access to all commands | `Ctrl+p` |
| 🎨 | **4 themes** — Monokai, Catppuccin, Solarized, High Contrast | `Ctrl+t` |

## ⌨️ Key Bindings

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `/` | Search | | `o` | Open in browser |
| `A` | Search arXiv API | | `P` | Open PDF |
| `j` / `k` | Navigate | | `d` | Download PDF |
| `Space` | Select | | `E` | Export menu |
| `s` | Cycle sort | | `c` | Copy to clipboard |
| `r` | Toggle read | | `Ctrl+s` | AI summary |
| `x` | Toggle star | | `C` | Chat with paper |
| `n` | Notes | | `L` | Relevance score |
| `p` | Abstract preview | | `Ctrl+p` | Command palette |
| `t` | Tags | | `G` | Citation graph |
| `V` | Check versions | | `R` | Recommendations |
| `Ctrl+b` | Save bookmark | | `1-9` | Jump to bookmark |
| `m` / `'` | Set / jump to mark | | `w` / `W` | Watch list |
| `Ctrl+k` | Collections | | `Ctrl+g` | Auto-tag (LLM) |
| `Ctrl+d` | Detail pane sections | | `Ctrl+h` | HuggingFace trending |
| `v` | Detail mode | | `Ctrl+Shift+b` | Remove bookmark |
| `Ctrl+e` | Toggle S2 / Exit API mode | | `?` | Help / shortcuts |

### Marks

Press `m` followed by a letter (`a`–`z`) to set a named mark at the current paper.
Press `'` followed by that letter to jump back to it. Marks persist within a session,
making it easy to navigate between papers you're comparing or revisiting.

## ⚙️ Configuration

Config lives at `~/.config/arxiv-browser/config.json` (Linux), `~/Library/Application Support/arxiv-browser/config.json` (macOS), or `%APPDATA%/arxiv-browser/config.json` (Windows).

See the **[full documentation](docs/)** for:

- 🤖 [AI summary & LLM setup](docs/llm-setup.md)
- 📊 [Semantic Scholar integration](docs/semantic-scholar.md)
- 🔥 [HuggingFace trending](docs/huggingface.md)
- 📅 [History mode & email ingestion](docs/history-mode.md)
- 🔍 [Search filters & advanced queries](docs/search-filters.md)
- 📥 [Export & PDF configuration](docs/export.md)
- 🛠️ [Troubleshooting](docs/troubleshooting.md)

## 🐚 Shell Completions

Enable tab completion for subcommands and flags:

```bash
# Bash (add to ~/.bashrc)
eval "$(arxiv-viewer completions bash)"

# Zsh (add to ~/.zshrc)
eval "$(arxiv-viewer completions zsh)"

# Fish (add to ~/.config/fish/config.fish)
arxiv-viewer completions fish | source
```

## 🛠️ Development

```bash
git clone https://github.com/NicolasSchuler/arxiv-subscription-viewer.git
cd arxiv-subscription-viewer && uv sync
just check   # lint + typecheck + tests
```

## 📄 License

MIT — [Nicolas Sebastian Schuler](mailto:nicolas.schuler@kit.edu)
