# 🗺️ Quick Reference / Cheat Sheet

A one-page reference for the most-used commands and keys. Press `?` in-app for the full,
always-current help overlay, and `Ctrl+p` for the command palette. To print the live binding
table from your installed version, run `arxiv-viewer keybindings`.

## CLI commands

```bash
arxiv-viewer                       # Browse newest history/ file (alias for `browse`)
arxiv-viewer search --category cs.AI            # Live arXiv search
arxiv-viewer search --query "diffusion" --field title
arxiv-viewer digest --category cs.AI --period weekly --output digest.md
arxiv-viewer digest --category cs.AI --tui      # Digest as an interactive inbox
arxiv-viewer dates                 # List local history dates
arxiv-viewer doctor                # Diagnose config, history, LLM, environment
arxiv-viewer config-path           # Print the config/cache directory
arxiv-viewer keybindings           # Print the key-binding reference
arxiv-viewer completions bash|zsh|fish          # Shell completions
```

## Navigate & triage

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `j` / `k` | Move down / up | | `T` | Quick triage visible unread |
| `Space` | Toggle selection | | `r` | Toggle read |
| `s` | Cycle sort order | | `x` | Toggle star |
| `[` / `]` | Previous / next date | | `Ctrl+r` | Mark visible as read |
| `m` / `'` | Set / jump to mark | | `1`–`9` | Jump to bookmark |

## Organize

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `t` | Tags | | `Ctrl+k` | Collections |
| `n` | Notes | | `w` / `W` | Watch filter / manage |
| `Ctrl+b` | Save bookmark | | `Ctrl+Shift+b` | Remove bookmark |

## Enrich (research tools)

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `Ctrl+s` | AI summary | | `e` / `Ctrl+e` | Fetch S2 / toggle S2 |
| `C` | Chat with paper | | `G` | Citation graph |
| `Ctrl+v` | Compare papers | | `R` | Recommendations |
| `L` | Relevance score | | `Ctrl+h` | HuggingFace trending |
| `Ctrl+g` | Auto-tag (LLM) | | `V` | Check versions |

## Read, export & view

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `o` | Open in browser | | `E` | Export menu |
| `P` | Open PDF in viewer | | `c` | Copy to clipboard |
| `d` | Download PDF | | `p` | Abstract preview |
| `F` | Preview PDF inline | | `v` | Detail mode |
| `I` | Preview first figure | | `z` | Compact list |
| `Alt+Left` / `Alt+Right` | Resize list/details | | `Alt+0` | Reset pane sizes |
| `y` | Read abstract aloud | | `Ctrl+t` | Cycle theme |

## Search syntax

Open search with `/`. Combine filters with `AND` / `OR` / `NOT` (precedence: `NOT` > `AND` > `OR`):

| Prefix | Matches |
|--------|---------|
| `cat:cs.AI` | Category |
| `author:hinton` | Author (use `author:=name` for exact) |
| `title:` / `abstract:` | Title / abstract text |
| `tag:to-read` | Tag |
| `unread` · `starred` · `review-due` | Virtual filters |
| `~ query` | Semantic (embedding) search |
| `@ query` | Run an arXiv API search |
| `> command` | Open the command palette |

See [search-filters.md](search-filters.md) for the full grammar.

---

**See also:** [Main README](../README.md) · [All Docs](README.md) · [Config Reference](config-reference.md)
