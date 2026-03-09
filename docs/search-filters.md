# 🔍 Search Filters & Advanced Queries

Press `/` to open the search box, `Escape` to cancel search or exit API mode. Combine any of these filters:

## Filter Prefixes

| Filter | Example | Description |
|--------|---------|-------------|
| `cat:` | `cat:cs.AI` | Filter by arXiv category |
| `tag:` | `tag:to-read` | Filter by custom tag |
| `author:` | `author:hinton` | Filter by author name |
| `title:` | `title:transformer` | Filter by title substring |
| `abstract:` | `abstract:attention` | Filter by abstract substring |
| `unread` | `unread` | Show only unread papers |
| `starred` | `starred` | Show only starred papers |
| (text) | `transformer` | Fuzzy search title/authors |
| `"..."` | `"large language"` | Match exact phrase |

## Boolean Operators

Combine terms with `AND`, `OR`, and `NOT`:

```
cat:cs.AI AND author:hinton
unread OR starred
NOT cat:math
cat:cs.LG AND title:transformer AND unread
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

## arXiv API Search

Press `A` to search all of arXiv via the API:

```bash
# CLI startup equivalents
arxiv-viewer --api-category cs.AI
arxiv-viewer --api-query "diffusion transformer" --api-field title
arxiv-viewer --api-query "agent benchmark" --api-category cs.LG
arxiv-viewer --api-query "transformer" --api-page-mode
```

API field options: `all`, `title`, `author`, `abstract`. Default page size is configurable:

```json
{
  "arxiv_api_max_results": 50
}
```

Values are clamped to the range `1..200`.
