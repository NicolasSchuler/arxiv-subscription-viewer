# 🔍 Search Filters & Advanced Queries

Press `/` to open the search box against your currently loaded dataset. Use the hint row for quick examples like `cat:cs.AI`, `author:hinton`, `unread`, or `"large language"`. Press `Escape` to close search or exit API/search-results mode.

## Filter Prefixes

| Filter | Example | Description |
|--------|---------|-------------|
| `cat:` | `cat:cs.AI` | Filter by arXiv category |
| `tag:` | `tag:to-read` | Filter by custom tag |
| `tag:` (quoted) | `tag:"to read"` | Match a tag value containing spaces |
| `author:` | `author:hinton` | Filter by author name |
| `author:` (quoted) | `author:"Geoffrey Hinton"` | Match a multi-word author substring |
| `title:` | `title:transformer` | Filter by title substring |
| `title:` (quoted) | `title:"large language"` | Match a multi-word title phrase |
| `abstract:` | `abstract:attention` | Filter by abstract substring |
| `abstract:` (quoted) | `abstract:"reinforcement learning"` | Match a multi-word abstract phrase |
| `unread` | `unread` | Show only unread papers |
| `starred` | `starred` | Show only starred papers |
| (text) | `transformer` | Fuzzy search title/authors |
| `"..."` | `"large language"` | Match a quoted phrase |

## Query Rules

- Adjacent terms imply `AND`: `cat:cs.AI unread` is the same as `cat:cs.AI AND unread`.
- Quoted values keep spaces inside one term: `author:"Geoffrey Hinton"`.
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
author:"Geoffrey Hinton" title:"neural networks"
cat:cs.AI unread NOT tag:archived
```

The last two examples rely on implicit `AND`, so they behave like:

```
author:"Geoffrey Hinton" AND title:"neural networks"
cat:cs.AI AND unread AND NOT tag:archived
```

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

---

**See also:** [LLM Setup](llm-setup.md) · [Config Reference](config-reference.md) · [Troubleshooting](troubleshooting.md) · [All Docs](README.md)
