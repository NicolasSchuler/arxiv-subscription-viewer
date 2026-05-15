# 🔍 Search Filters & Advanced Queries

Press `/` to open the search box against your currently loaded dataset. Use the hint row for quick examples like `cat:cs.AI`, `author:hinton`, `review-due`, or `"large language"`. Press `Escape` to close search or exit API/search-results mode.

## Filter Prefixes

| Filter | Example | Description |
|--------|---------|-------------|
| `cat:` | `cat:cs.AI` | Filter by arXiv category |
| `tag:` | `tag:to-read` | Filter by custom tag |
| `tag:` (quoted) | `tag:"to read"` | Match a tag value containing spaces |
| `author:` | `author:hinton` | Match an author substring |
| `author:` (quoted) | `author:"Geoffrey Hinton"` | Match a multi-word author substring |
| `author:=` | `author:=hinton` | Exact-match a single-token author name |
| `author:=` (quoted) | `author:="Geoffrey Hinton"` | Exact-match a full author name |
| `title:` | `title:transformer` | Filter by title substring |
| `title:` (quoted) | `title:"large language"` | Match a multi-word title phrase |
| `abstract:` | `abstract:attention` | Filter by abstract substring |
| `abstract:` (quoted) | `abstract:"reinforcement learning"` | Match a multi-word abstract phrase |
| `unread` | `unread` | Show only unread papers |
| `starred` | `starred` | Show only starred papers |
| `review-due` | `review-due` | Show papers whose next spaced-review date is today or earlier |
| (text) | `transformer` | Fuzzy search title/authors |
| `~` | `~ papers about hallucination in RAG` | Semantic search over titles and abstracts when an embedding backend is available |
| `"..."` | `"large language"` | Match a quoted phrase |

## Query Rules

- Adjacent terms imply `AND`: `cat:cs.AI unread` is the same as `cat:cs.AI AND unread`.
- Quoted values keep spaces inside one term: `author:"Geoffrey Hinton"`.
- `author:` is substring-based for quick surname searches; `author:=` is exact after normalizing case, punctuation, and whitespace.
- Unquoted field values stop at the next space, so use quotes when a field value contains spaces.
- Operator precedence is `NOT` > `AND` > `OR`.
- Parentheses are not supported.

## Boolean Operators

Combine terms with `AND`, `OR`, and `NOT`:

```
cat:cs.AI AND author:hinton
unread OR starred
NOT cat:math
cat:cs.LG AND title:transformer AND unread
review-due AND unread
review-due AND cat:cs.LG
author:"Geoffrey Hinton" title:"neural networks"
author:="Geoffrey Hinton" AND unread
cat:cs.AI unread NOT tag:archived
```

The last two examples rely on implicit `AND`, so they behave like:

```
author:"Geoffrey Hinton" AND title:"neural networks"
cat:cs.AI AND unread AND NOT tag:archived
```

## Author Tracking

Use `Ctrl+p` → **Author Profile** on a paper to see all loaded papers for one author, co-author counts, and cached Semantic Scholar citation counts when already available. Use `Ctrl+p` → **Track Author** to add an exact author name to `tracked_authors`; matching papers are highlighted like watch-list hits.

| Key | Action |
|-----|--------|
| `/` | Open search box |
| `Escape` | Cancel search / exit API mode |

## Search Bookmarks

Save up to 9 frequent searches for quick access:

| Key | Action |
|-----|--------|
| `Ctrl+b` | Save current search as bookmark |
| `1`-`9` | Jump to saved bookmark |
| `Ctrl+Shift+b` | Remove active bookmark |

### Watch Lists vs Bookmarks

- **Watch lists** (`w`/`W`): Highlight papers matching author/keyword/title patterns across all views. Use these to track topics or researchers you always want to notice.
- **Bookmarks** (`1-9`, `Ctrl+b`): Save specific search queries for quick recall. Use these for complex filters you run frequently.
- **Review due** (`review-due`): Show papers you explicitly scheduled for spaced review. Use `Ctrl+p` to schedule, mark reviewed, clear review, or jump straight to due reviews.

## arXiv API Search

Press `A` in the app, or start from the CLI, to search all of arXiv via the API:

```bash
# CLI startup equivalents
arxiv-viewer search --category cs.AI
arxiv-viewer search --query "diffusion transformer" --field title
arxiv-viewer search --query "agent benchmark" --category cs.LG
arxiv-viewer search --query "transformer" --mode page
```

API field options: `all`, `title`, `author`, `abstract`. Default page size is configurable:

```json
{
  "arxiv_api_max_results": 50
}
```

Values are clamped to the range `1..200`.

If API search or paging behaves unexpectedly, run `arxiv-viewer doctor` from your usual launch directory to confirm whether the app sees local `history/` files or is operating purely in live-search mode.

## Semantic Search

Prefix a local query with `~` to rank loaded papers by local text embeddings:

```text
~ papers about reducing hallucinations in RAG systems
```

Install the lightweight backend with:

```bash
pip install "arxiv-subscription-viewer[semantic-fastembed]"
```

For larger Hugging Face models such as Qwen, install `sentence-transformers` separately and configure:

```json
{
  "semantic_search_backend": "sentence-transformers",
  "semantic_search_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

For a local OpenAI-compatible embeddings server, set:

```json
{
  "semantic_search_backend": "http",
  "semantic_search_api_base_url": "http://localhost:8080",
  "semantic_search_model": "Qwen/Qwen3-Embedding-0.6B"
}
```

If semantic search is not configured or the model cannot load, the app strips the `~` prefix and falls back to the existing fuzzy search.

---

**See also:** [LLM Setup](llm-setup.md) · [Config Reference](config-reference.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
